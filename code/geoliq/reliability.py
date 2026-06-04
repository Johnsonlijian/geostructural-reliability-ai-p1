"""Reliability toolkit: bootstrap AUC CIs, cluster-bootstrap paired AUC tests,
calibration (Brier/ECE), and split-conformal coverage under random vs grouped shift.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(y, s):
    return roc_auc_score(y, s)


def bootstrap_auc_ci(y, score, n_boot=1000, groups=None, seed=0, alpha=0.05):
    y = np.asarray(y)
    score = np.asarray(score, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    if groups is None:
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) > 1:
                aucs.append(_auc(y[idx], score[idx]))
    else:
        groups = np.asarray(groups)
        gids = np.unique(groups)
        members = {gg: np.where(groups == gg)[0] for gg in gids}
        for _ in range(n_boot):
            sg = rng.choice(gids, len(gids), replace=True)
            idx = np.concatenate([members[gg] for gg in sg])
            if len(np.unique(y[idx])) > 1:
                aucs.append(_auc(y[idx], score[idx]))
    aucs = np.array(aucs)
    return {"auc": float(_auc(y, score)),
            "lo": float(np.percentile(aucs, 100 * alpha / 2)),
            "hi": float(np.percentile(aucs, 100 * (1 - alpha / 2)))}


def paired_auc_diff_ci(y, score_a, score_b, n_boot=1000, groups=None, seed=0, alpha=0.05):
    """Delta = AUC(a) - AUC(b); cluster-bootstrap if groups given. Returns CI + P(Delta>0)."""
    y = np.asarray(y)
    a = np.asarray(score_a, float)
    b = np.asarray(score_b, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    if groups is not None:
        groups = np.asarray(groups)
        gids = np.unique(groups)
        members = {gg: np.where(groups == gg)[0] for gg in gids}
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            sg = rng.choice(gids, len(gids), replace=True)
            idx = np.concatenate([members[gg] for gg in sg])
        if len(np.unique(y[idx])) > 1:
            diffs.append(_auc(y[idx], a[idx]) - _auc(y[idx], b[idx]))
    diffs = np.array(diffs)
    return {"delta": float(_auc(y, a) - _auc(y, b)),
            "lo": float(np.percentile(diffs, 100 * alpha / 2)),
            "hi": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
            "P(delta>0)": float(np.mean(diffs > 0))}


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    n = len(y)
    e = 0.0
    for i in range(bins):
        hi = p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return float(e)


def split_conformal_sets(p1_cal, y_cal, p1_test, alpha=0.1):
    """Split-conformal binary prediction sets (marginal coverage 1-alpha)."""
    p1_cal = np.asarray(p1_cal, float)
    y_cal = np.asarray(y_cal, int)
    s_cal = np.where(y_cal == 1, 1 - p1_cal, p1_cal)  # nonconformity = 1 - p_truelabel
    n = len(s_cal)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    q = np.sort(s_cal)[k - 1]
    p1_test = np.asarray(p1_test, float)
    inc1 = (1 - p1_test) <= q
    inc0 = p1_test <= q
    return inc0, inc1


def conformal_coverage_experiment(X, y, groups, model_factory, mode="random",
                                  alpha=0.1, n_rep=12, seed=0):
    """Mean empirical coverage of split-conformal sets, random vs group-held-out calibration."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    n = len(y)
    covs, effs = [], []
    for _ in range(n_rep):
        if mode == "random":
            perm = rng.permutation(n)
            ntr, nca = int(0.4 * n), int(0.3 * n)
            tr, ca, te = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
        else:
            gids = rng.permutation(np.unique(groups))
            ng = len(gids)
            gtr, gca, gte = gids[:int(0.4 * ng)], gids[int(0.4 * ng):int(0.7 * ng)], gids[int(0.7 * ng):]
            tr = np.where(np.isin(groups, gtr))[0]
            ca = np.where(np.isin(groups, gca))[0]
            te = np.where(np.isin(groups, gte))[0]
        if len(np.unique(y[tr])) < 2 or len(ca) < 5 or len(te) < 5:
            continue
        mdl = model_factory()
        mdl.fit(X[tr], y[tr])
        p_ca = mdl.predict_proba(X[ca])[:, 1]
        p_te = mdl.predict_proba(X[te])[:, 1]
        inc0, inc1 = split_conformal_sets(p_ca, y[ca], p_te, alpha)
        covered = np.where(y[te] == 1, inc1, inc0)
        covs.append(float(covered.mean()))
        effs.append(float((inc0.astype(int) + inc1.astype(int)).mean()))
    return {"mode": mode, "target_coverage": 1 - alpha, "n_rep_used": len(covs),
            "mean_coverage": float(np.mean(covs)) if covs else float("nan"),
            "std_coverage": float(np.std(covs)) if covs else float("nan"),
            "mean_set_size": float(np.mean(effs)) if effs else float("nan")}


def conformal_conditional_coverage(X, y, groups, model_factory, alpha=0.1, n_rep=12, seed=0, min_pts=3):
    """Per-held-out-earthquake (conditional) coverage under group-held-out conformal calibration.

    Split-conformal guarantees MARGINAL coverage; this measures whether any single new
    earthquake actually attains it. Reports the fraction of earthquakes that are under-covered.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    pgc = []
    for _ in range(n_rep):
        gids = rng.permutation(np.unique(groups))
        ng = len(gids)
        gtr, gca, gte = gids[:int(0.4 * ng)], gids[int(0.4 * ng):int(0.7 * ng)], gids[int(0.7 * ng):]
        tr = np.where(np.isin(groups, gtr))[0]
        ca = np.where(np.isin(groups, gca))[0]
        te = np.where(np.isin(groups, gte))[0]
        if len(np.unique(y[tr])) < 2 or len(ca) < 5 or len(te) < 5:
            continue
        mdl = model_factory()
        mdl.fit(X[tr], y[tr])
        inc0, inc1 = split_conformal_sets(mdl.predict_proba(X[ca])[:, 1], y[ca],
                                          mdl.predict_proba(X[te])[:, 1], alpha)
        covered = np.where(y[te] == 1, inc1, inc0)
        gte_arr = groups[te]
        for gg in np.unique(gte_arr):
            m = gte_arr == gg
            if m.sum() >= min_pts:
                pgc.append(float(covered[m].mean()))
    pgc = np.array(pgc)
    return {"target_coverage": 1 - alpha, "n_event_evals": int(len(pgc)),
            "mean_conditional_coverage": float(np.mean(pgc)),
            "p10_conditional_coverage": float(np.percentile(pgc, 10)),
            "min_conditional_coverage": float(np.min(pgc)),
            "frac_events_undercovered": float(np.mean(pgc < (1 - alpha)))}
