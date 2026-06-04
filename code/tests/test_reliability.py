import numpy as np
from geoliq import reliability as R


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    good = np.clip(y + rng.normal(0, 0.5, n), 0, 1)   # informative score
    bad = rng.uniform(0, 1, n)                          # random score
    return y, good, bad


def test_bootstrap_ci_brackets_point():
    y, good, _ = _toy()
    ci = R.bootstrap_auc_ci(y, good, n_boot=300)
    assert ci["lo"] <= ci["auc"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] > 0


def test_paired_diff_detects_better_model():
    y, good, bad = _toy()
    d = R.paired_auc_diff_ci(y, good, bad, n_boot=300)
    assert d["delta"] > 0
    assert d["P(delta>0)"] > 0.9


def test_brier_and_ece_perfect():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert R.brier(y, y.astype(float)) == 0.0
    assert R.ece(y, y.astype(float)) < 1e-9


def test_conformal_random_coverage_near_target():
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(0, 1, (n, 3))
    y = (X[:, 0] + 0.5 * rng.normal(0, 1, n) > 0).astype(int)
    groups = rng.integers(0, 20, n)
    out = R.conformal_coverage_experiment(X, y, groups, lambda: HistGradientBoostingClassifier(random_state=0),
                                          mode="random", alpha=0.1, n_rep=8)
    assert 0.80 <= out["mean_coverage"] <= 1.0  # random-split conformal ~ valid (>= ~target)
