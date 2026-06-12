"""Two-axis Vs critical-band closure pipeline.

Run:  python run_closure.py
Input: a machine-readable KEA2013 Table S1 (xlsx/xls/csv) dropped into ./incoming/
       (filename containing 'table_s1' or 'supplemental'). If only a PDF table is
       available, extract it to CSV first and drop the CSV.
Everything is deterministic (fixed seeds). No mechanism coefficient is refit.

Outputs in this folder:
  e0_inventory.json / e0_status.json
  e1_results.json
  e2_pairing_audit.csv / e2_results.json
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vs_kayen2013 as vk  # noqa: E402

MODULE = Path(__file__).resolve().parent
CODE_ROOT = MODULE.parent
INCOMING = MODULE / "incoming"
RESULTS = MODULE
CETIN_RECORDS = CODE_ROOT / "data" / "processed" / "cetin2018_baseline_records.csv"

SEED = 7
N_BOOT = 2000
PRACTICAL_DAUC = 0.02
PRACTICAL_DNLL = 0.05
BAND = (0.3, 0.7)

# Candidate column-name fragments for the Table S1 schema (matched lowercase).
SCHEMA = {
    "vs1": ["vs1", "v s1", "v_s1", "vs,1", "vs1 (m/s)"],
    "csr": ["csr"],
    "mw": ["mw", "magnitude", "m w"],
    "sigma_eff": ["sigma", "effective", "s'v", "σ'", "svo'", "s 'vo"],
    "fc": ["fc", "fines"],
    "label": ["liquef", "performance", "observ", "class"],
    "earthquake": ["earthquake", "event", "eq"],
    "site": ["site", "location", "name"],
    "amax": ["amax", "a max", "pga"],
    "gwt": ["gwt", "water"],
    "depth": ["depth", "dcr"],
}

YES = {"yes", "y", "liq", "liquefaction", "1", "true", "l"}
NO = {"no", "n", "nonliq", "no liquefaction", "non-liquefaction", "0", "false", "nl"}
MARGINAL = {"marginal", "m", "yes/no", "marg"}


def log(msg: str) -> None:
    print(f"[closure] {msg}")


def find_table_s1() -> Path | None:
    pats = ("table_s1", "tables1", "supplemental")
    for ext in (".xlsx", ".xls", ".csv"):
        for f in sorted(INCOMING.glob(f"*{ext}")):
            if any(p in f.name.lower() for p in pats):
                return f
    return None


def map_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    cols = {c: str(c).strip().lower() for c in df.columns}
    for key, frags in SCHEMA.items():
        for c, cl in cols.items():
            if any(fr in cl for fr in frags) and c not in mapping.values():
                mapping[key] = c
                break
    return mapping


def normalize_label(v) -> float | None:
    s = str(v).strip().lower()
    if s in YES:
        return 1.0
    if s in NO:
        return 0.0
    if s in MARGINAL:
        return None
    try:
        f = float(s)
        return f if f in (0.0, 1.0) else None
    except ValueError:
        return None


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def eq_year_tokens(s: str):
    s = str(s)
    year = None
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        year = m.group(0)
    toks = set(re.findall(r"[a-z]{3,}", s.lower())) - {"earthquake", "the", "and"}
    return year, toks


# ----------------------------------------------------------------- E0

def e0(df: pd.DataFrame, mapping: dict, src: Path) -> dict:
    need = ["vs1", "csr", "mw", "sigma_eff", "label", "earthquake"]
    missing = [k for k in need if k not in mapping]
    info = {
        "source_file": src.name,
        "n_rows_raw": int(len(df)),
        "columns_found": {k: str(v) for k, v in mapping.items()},
        "columns_missing": missing,
    }
    if missing:
        info["status"] = "FAIL_SCHEMA"
        return info
    lab = df[mapping["label"]].map(normalize_label)
    usable = df[lab.notna()].copy()
    usable["__y"] = lab[lab.notna()].astype(float)
    n_events = usable[mapping["earthquake"]].nunique()
    info.update(
        n_marginal_excluded=int((lab.isna()).sum()),
        n_usable=int(len(usable)),
        n_events=int(n_events),
        n_liq=int(usable["__y"].sum()),
        n_noliq=int((1 - usable["__y"]).sum()),
        fc_coverage=float(usable[mapping["fc"]].notna().mean()) if "fc" in mapping else 0.0,
        status="PASS" if len(usable) >= 250 and n_events >= 12 else "FAIL_POWER",
    )
    return info


# ----------------------------------------------------------- E1 helpers

def oof_probs(model_fn, X, y, groups, seed=SEED):
    """Out-of-fold probabilities under event-grouped CV."""
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    p = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "nll": float(log_loss(y, p))}


def event_bootstrap_delta(y, p_a, p_b, groups, n_boot=N_BOOT, seed=SEED):
    """Paired cluster bootstrap over events for AUC(b)-AUC(a) and NLL(a)-NLL(b)."""
    rng = np.random.default_rng(seed)
    ev = np.unique(groups)
    idx_by_ev = {e: np.where(groups == e)[0] for e in ev}
    d_auc, d_nll = [], []
    for _ in range(n_boot):
        take = rng.choice(ev, size=len(ev), replace=True)
        idx = np.concatenate([idx_by_ev[e] for e in take])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        pa, pb = p_a[idx], p_b[idx]
        d_auc.append(roc_auc_score(yy, pb) - roc_auc_score(yy, pa))
        d_nll.append(log_loss(yy, pa) - log_loss(yy, pb))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        "d_auc": float(np.median(d_auc)), "d_auc_ci": q(d_auc),
        "d_nll": float(np.median(d_nll)), "d_nll_ci": q(d_nll),
        "n_boot_effective": len(d_auc),
    }


def learner_zoo(n_features, mono_cst=None):
    zoo = {
        "full_logistic": lambda: LogisticRegression(max_iter=2000),
        "random_forest": lambda: RandomForestClassifier(n_estimators=400, random_state=0),
        "hgb": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    if mono_cst is not None:
        zoo["hgb_monotonic"] = lambda: HistGradientBoostingClassifier(
            random_state=0, monotonic_cst=mono_cst)
    return zoo


def e1(df: pd.DataFrame, mapping: dict) -> dict:
    m = mapping
    lab = df[m["label"]].map(normalize_label)
    d = df[lab.notna()].copy()
    y = lab[lab.notna()].astype(float).to_numpy()
    groups = d[m["earthquake"]].astype(str).to_numpy()

    fc = pd.to_numeric(d[m["fc"]], errors="coerce") if "fc" in m else pd.Series(0.0, index=d.index)
    fc_filled = fc.fillna(0.0).to_numpy()
    vs1 = pd.to_numeric(d[m["vs1"]], errors="coerce").to_numpy()
    csr = pd.to_numeric(d[m["csr"]], errors="coerce").to_numpy()
    mw = pd.to_numeric(d[m["mw"]], errors="coerce").to_numpy()
    sig = pd.to_numeric(d[m["sigma_eff"]], errors="coerce").to_numpy()
    ok = np.isfinite(vs1) & np.isfinite(csr) & np.isfinite(mw) & np.isfinite(sig) & (csr > 0) & (sig > 0)
    vs1, csr, mw, sig, fc_filled, y, groups = (a[ok] for a in (vs1, csr, mw, sig, fc_filled, y, groups))
    log(f"E1 usable complete-feature rows: {ok.sum()} / {len(ok)}")

    s_vs = vk.margin(vs1, csr, mw, sig, fc_filled)

    # zero-shot margin discrimination (no fitting at all)
    res = {"n": int(len(y)), "n_events": int(len(np.unique(groups))),
           "margin_zero_shot": metrics(y, np.clip(norm.cdf(s_vs), 1e-6, 1 - 1e-6))}

    feat_cols = [vs1, csr, mw, sig, fc_filled]
    feat_names = ["vs1", "csr", "mw", "sigma_eff", "fc"]
    extra = []
    for k in ("amax", "gwt", "depth"):
        if k in m:
            v = pd.to_numeric(df.loc[df.index[lab.notna()], m[k]], errors="coerce").to_numpy()[ok]
            if np.isfinite(v).all():
                feat_cols.append(v); feat_names.append(k); extra.append(k)
    X_full = np.column_stack(feat_cols)
    X_margin = s_vs.reshape(-1, 1)
    X_fusion = np.column_stack([X_full, s_vs])
    mono = [-1, 1, 1, 0, -1] + [0] * len(extra)

    p_margin = oof_probs(lambda: LogisticRegression(max_iter=2000), X_margin, y, groups)
    res["margin_only_logistic"] = metrics(y, p_margin)

    cand = {}
    for name, fn in learner_zoo(X_full.shape[1], mono).items():
        cand[name] = oof_probs(fn, X_full, y, groups)
        res[name] = metrics(y, cand[name])
    cand["fusion_hgb"] = oof_probs(lambda: HistGradientBoostingClassifier(random_state=0),
                                   X_fusion, y, groups)
    res["fusion_hgb"] = metrics(y, cand["fusion_hgb"])

    best = max(cand, key=lambda k: res[k]["auc"])
    res["best_nonmargin"] = best
    res["paired_vs_margin"] = event_bootstrap_delta(y, p_margin, cand[best], groups)
    res["practical_gain_excluded"] = bool(
        res["paired_vs_margin"]["d_auc_ci"][1] < PRACTICAL_DAUC
        and res["paired_vs_margin"]["d_nll_ci"][1] < PRACTICAL_DNLL)

    # random-split optimism (100 repeats, row-level stratified 5-fold)
    rng = np.random.default_rng(SEED)
    rand_auc = []
    for r in range(100):
        skf = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e9)))
        p = np.full(len(y), np.nan)
        for tr, te in skf.split(X_full, y):
            mm = HistGradientBoostingClassifier(random_state=0)
            mm.fit(X_full[tr], y[tr])
            p[te] = mm.predict_proba(X_full[te])[:, 1]
        rand_auc.append(roc_auc_score(y, p))
    res["random_split_hgb_auc_median"] = float(np.median(rand_auc))
    res["random_minus_grouped_optimism"] = float(np.median(rand_auc) - res["hgb"]["auc"])
    return res


# ----------------------------------------------------------- E2 pairing

def e2_pair(df_vs: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    cet = pd.read_csv(CETIN_RECORDS)
    cet["site_key"] = cet["site"].map(norm_key)
    cet["eq_year"] = cet["earthquake"].map(lambda s: eq_year_tokens(s)[0])

    lab = df_vs[mapping["label"]].map(normalize_label)
    dv = df_vs[lab.notna()].copy()
    dv["__y"] = lab[lab.notna()].astype(float)
    site_col = mapping.get("site")
    if site_col is None:
        return pd.DataFrame()
    dv["site_key"] = dv[site_col].map(norm_key)
    dv["eq_year"] = dv[mapping["earthquake"]].map(lambda s: eq_year_tokens(s)[0])

    def site_sim(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if min(len(a), len(b)) >= 5 and (a in b or b in a):
            return 0.95
        return difflib.SequenceMatcher(None, a, b).ratio()

    rows = []
    for _, r in dv.iterrows():
        sub = cet[cet["eq_year"] == r["eq_year"]]
        if sub.empty or not r["site_key"]:
            continue
        ratios = sub["site_key"].map(lambda k: site_sim(k, r["site_key"]))
        j = ratios.idxmax()
        if ratios[j] >= 0.85:
            c = cet.loc[j]
            rows.append({
                "vs_site": r[site_col], "cetin_site": c["site"],
                "vs_eq": r[mapping["earthquake"]], "cetin_eq": c["earthquake"],
                "match_ratio": float(ratios[j]),
                "y_vs": float(r["__y"]), "y_cetin": float(c["y"]),
                "s_pen": float(c["score_liq"]),
                "vs1": float(pd.to_numeric(r[mapping["vs1"]], errors="coerce")),
                "csr": float(pd.to_numeric(r[mapping["csr"]], errors="coerce")),
                "mw": float(pd.to_numeric(r[mapping["mw"]], errors="coerce")),
                "sigma_eff": float(pd.to_numeric(r[mapping["sigma_eff"]], errors="coerce")),
                "fc": float(pd.to_numeric(r[mapping["fc"]], errors="coerce")) if "fc" in mapping else np.nan,
                "earthquake": str(c["earthquake"]),
            })
    return pd.DataFrame(rows)


def e2(paired: pd.DataFrame) -> dict:
    res = {"n_pairs": int(len(paired)),
           "n_events": int(paired["earthquake"].nunique()) if len(paired) else 0}
    res["label_agreement"] = float((paired["y_vs"] == paired["y_cetin"]).mean()) if len(paired) else None
    if res["n_pairs"] < 60 or res["n_events"] < 8:
        res["status"] = "UNPOWERED"
        return res

    d = paired.dropna(subset=["vs1", "csr", "mw", "sigma_eff", "s_pen"]).copy()
    d["fc"] = d["fc"].fillna(0.0)
    y = d["y_cetin"].to_numpy()
    groups = d["earthquake"].astype(str).to_numpy()
    s_vs = vk.margin(d["vs1"].to_numpy(), d["csr"].to_numpy(), d["mw"].to_numpy(),
                     d["sigma_eff"].to_numpy(), d["fc"].to_numpy())
    s_pen = d["s_pen"].to_numpy()

    XA = s_pen.reshape(-1, 1)
    XB = np.column_stack([s_pen, s_vs])
    pA = oof_probs(lambda: LogisticRegression(max_iter=2000), XA, y, groups)
    pB = oof_probs(lambda: LogisticRegression(max_iter=2000), XB, y, groups)
    res["pen_only"] = metrics(y, pA)
    res["pen_plus_vs"] = metrics(y, pB)
    res["paired_overall"] = event_bootstrap_delta(y, pA, pB, groups)

    inband = (pA >= BAND[0]) & (pA <= BAND[1])
    res["n_inband"] = int(inband.sum())
    if inband.sum() >= 20 and len(np.unique(y[inband])) == 2:
        res["inband_pen_only"] = metrics(y[inband], pA[inband])
        res["inband_pen_plus_vs"] = metrics(y[inband], pB[inband])
        res["paired_inband"] = event_bootstrap_delta(
            y[inband], pA[inband], pB[inband], groups[inband])
        ci = res["paired_inband"]["d_nll_ci"]
        res["H1_supported"] = bool(ci[0] > 0 and res["paired_inband"]["d_nll"] > PRACTICAL_DNLL)
    else:
        res["inband_note"] = "in-band subset too small for a powered closure readout"
        res["H1_supported"] = None
    res["status"] = "DONE"
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    src = find_table_s1()
    if src is None:
        status = {"status": "blocked",
                  "reason": "KEA2013 Table S1 not found in incoming/ "
                            "(need *.xlsx/*.xls/*.csv with 'table_s1' or 'supplemental' in name)"}
        (RESULTS / "e0_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        log("E0 BLOCKED: drop Table S1 into incoming/ and re-run.")
        return
    log(f"loading {src.name}")
    if src.suffix == ".csv":
        df = pd.read_csv(src)
    else:
        df = pd.read_excel(src)
    mapping = map_columns(df)
    inv = e0(df, mapping, src)
    (RESULTS / "e0_inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    log(f"E0: {inv['status']} ({inv.get('n_usable', 0)} usable, {inv.get('n_events', 0)} events)")
    if inv["status"] != "PASS":
        return

    res1 = e1(df, mapping)
    (RESULTS / "e1_results.json").write_text(json.dumps(res1, indent=2), encoding="utf-8")
    log(f"E1: margin zero-shot AUC={res1['margin_zero_shot']['auc']:.3f}; "
        f"best learner {res1['best_nonmargin']} dAUC={res1['paired_vs_margin']['d_auc']:+.3f} "
        f"{res1['paired_vs_margin']['d_auc_ci']}")

    paired = e2_pair(df, mapping)
    paired.to_csv(RESULTS / "e2_pairing_audit.csv", index=False)
    res2 = e2(paired)
    (RESULTS / "e2_results.json").write_text(json.dumps(res2, indent=2), encoding="utf-8")
    log(f"E2: {res2['status']}, pairs={res2['n_pairs']}, events={res2['n_events']}, "
        f"H1={res2.get('H1_supported')}")


if __name__ == "__main__":
    main()
