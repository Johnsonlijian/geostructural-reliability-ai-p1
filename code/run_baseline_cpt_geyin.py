"""(3-real) Cross-database replication on the INDEPENDENT CPT DB (Geyin & Maurer 2021).

Zero-shot BI2014-CPT triggering over the profile -> LPI (Iwasaki 1978) as the
triggering->surface-manifestation bridge, vs observed manifestation; plus
random-vs-earthquake-grouped optimism gap. Compare physics across the two global DBs.
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import mat73
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logging.disable(logging.ERROR)
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from geoliq.mechanics import cpt_profile as CP

PATH = os.path.join(BASE, "data", "raw", "geyin_maurer_2021_cpt", "GLOBALDATASETV1.mat")
D = mat73.loadmat(PATH)["GLOBALDATASET"]


def g(c, k):
    return float(np.asarray(c[k]).ravel()[0])


rows = []
for c in D:
    res = CP.process_case(np.asarray(c["depth"]), np.asarray(c["qc_inv"]), np.asarray(c["fs_inv"]),
                          g(c, "GWT"), g(c, "Magnitude"), g(c, "PGA"))
    rows.append({"event": c["EventName"], "Mw": g(c, "Magnitude"), "PGA": g(c, "PGA"), "GWT": g(c, "GWT"),
                 "y": 1 if g(c, "Manifestation") == 2 else 0, **res})
df = pd.DataFrame(rows)
df["crit_FS_capped"] = np.where(np.isfinite(df["crit_FS"]), df["crit_FS"], 5.0)
df.to_csv(os.path.join(BASE, "data", "processed", "geyin2021_cpt_records.csv"), index=False)

y = df["y"].to_numpy()
# Physics manifestation baseline = LPI (Iwasaki 1978); deterministic threshold LPI>=5
auc_phys = float(roc_auc_score(y, df["LPI"].to_numpy()))
pred = (df["LPI"].to_numpy() >= 5.0).astype(int)
acc = float(accuracy_score(y, pred))
cm = confusion_matrix(y, pred)
tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

FEAT = ["Mw", "PGA", "GWT", "LPI", "crit_FS_capped", "crit_depth", "qc1Ncs_crit", "Ic_crit", "sev_crit"]
X = df[FEAT].to_numpy(float)
groups = df["event"].to_numpy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
gkf = list(GroupKFold(n_splits=5).split(X, y, groups))
hgb = lambda: HistGradientBoostingClassifier(random_state=0)
logit = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))


def oof(estf, cv):
    return float(roc_auc_score(y, cross_val_predict(estf(), X, y, cv=cv, method="predict_proba")[:, 1]))


ml = {"hist_gbt": {"random": round(oof(hgb, skf), 4), "event_grouped": round(oof(hgb, gkf), 4)},
      "logistic": {"random": round(oof(logit, skf), 4), "event_grouped": round(oof(logit, gkf), 4)}}
for k in ml:
    ml[k]["optimism_gap"] = round(ml[k]["random"] - ml[k]["event_grouped"], 4)

out = {
    "dataset": "Geyin & Maurer (2021) CPT, DOI 10.17603/ds2-wftt-mv37, PRJ-3012",
    "n_cases": int(len(df)), "n_events": int(df["event"].nunique()),
    "n_yes": int((df["y"] == 1).sum()), "n_no": int((df["y"] == 0).sum()),
    "physics_baseline": "BI2014-CPT triggering -> LPI (Iwasaki 1978)",
    "BI2014_CPT_LPI_zero_shot": {"roc_auc_LPI": round(auc_phys, 4), "accuracy_LPI>=5": round(acc, 4),
                                 "confusion_[[TN,FP],[FN,TP]]": [[tn, fp], [fn, tp]],
                                 "false_alarm": round(fp / (fp + tn), 4), "miss": round(fn / (fn + tp), 4)},
    "ML_optimism_gap": ml,
    "cross_DB": {"SPT_Cetin2018_physics_auc_triggering": 0.923,
                 "CPT_Geyin2021_physics_auc_manifestation_LPI": round(auc_phys, 4)},
}
with open(os.path.join(BASE, "data", "processed", "geyin2021_cpt_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("=" * 72)
print("(3-REAL) CPT cross-DB - Geyin & Maurer (2021), zero-shot BI2014-CPT + LPI")
print("=" * 72)
print(f"n={len(df)} cases / {df['event'].nunique()} earthquakes; yes={int((df['y']==1).sum())} no={int((df['y']==0).sum())}")
print(f"BI2014-CPT + LPI(Iwasaki) vs manifestation: AUC(LPI)={auc_phys:.3f}  acc(LPI>=5)={acc:.3f}")
print(f"   confusion [[TN={tn},FP={fp}],[FN={fn},TP={tp}]]  FAR={fp/(fp+tn):.3f}  miss={fn/(fn+tp):.3f}")
print("ML optimism gap (random -> event-grouped):")
for k, v in ml.items():
    print(f"   {k:<10} random={v['random']:.3f}  grouped={v['event_grouped']:.3f}  gap={v['optimism_gap']:.3f}")
print(f"\nCROSS-DB physics (zero-shot, no fit): SPT-Cetin triggering AUC=0.923  |  CPT-Geyin manifestation(LPI) AUC={auc_phys:.3f}")
print("wrote: data/processed/geyin2021_cpt_records.csv + geyin2021_cpt_results.json")
