"""Zero-shot Vs liquefaction mechanism engine.

Source: Kayen et al. (2013), J. Geotech. Geoenviron. Eng. 139(3):407-419,
Eqs. (13)-(17) transcribed from `../incoming/kayen2013_asce_official.pdf`
(pp. 413-414), with the Eq. (14) sign correction from Kayen et al. (2015),
Erratum, JGGE 141(9):08215002, DOI 10.1061/(ASCE)GT.1943-5606.0001390
(`../incoming/kayen2015_erratum.pdf`): the Phi^-1(P_L) term is +0.4809, which
makes Eq. (14) the exact inverse of Eq. (13).

Hard rules (IMUT zero-shot): the published coefficients below are constants and
are never refit on labels. sigma'_v0 enters in kPa (paper text reports the
catalog mean as 60.2 kPa); confirm against the Table S1 column header on load.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Published limit-state coefficients, Kayen et al. (2013) Eq. (13).
TH1 = 0.0073   # Vs1 scale
TH2 = 2.8011   # Vs1 exponent
TH3 = 1.946    # ln(CSR)
TH4 = 2.6168   # ln(Mw)
TH5 = 0.0099   # ln(sigma'_v0)
TH6 = 0.0028   # FC
SIGMA_EPS = 0.4809  # model error


def limit_state(vs1, csr, mw, sigma_eff_kpa, fc=0.0):
    """Eq. (13) bracket term. Positive = stable side, negative = liquefaction side."""
    vs1 = np.asarray(vs1, dtype=float)
    csr = np.asarray(csr, dtype=float)
    mw = np.asarray(mw, dtype=float)
    sigma_eff_kpa = np.asarray(sigma_eff_kpa, dtype=float)
    fc = np.asarray(fc, dtype=float)
    return ((TH1 * vs1) ** TH2
            - TH3 * np.log(csr)
            - TH4 * np.log(mw)
            - TH5 * np.log(sigma_eff_kpa)
            + TH6 * fc)


def margin(vs1, csr, mw, sigma_eff_kpa, fc=0.0):
    """Standardized Vs margin s_Vs = -g/sigma_eps, so P_L = Phi(s_Vs).

    Higher s_Vs = closer to / deeper into liquefaction. This is the
    non-refitted mechanism coordinate used by the MAV-CP audit.
    """
    return -limit_state(vs1, csr, mw, sigma_eff_kpa, fc) / SIGMA_EPS


def p_liq(vs1, csr, mw, sigma_eff_kpa, fc=0.0):
    """Eq. (13): probability of liquefaction."""
    return norm.cdf(margin(vs1, csr, mw, sigma_eff_kpa, fc))


def crr(vs1, mw, sigma_eff_kpa, fc=0.0, pl=0.5):
    """Eq. (14) with the 2015 erratum sign (+SIGMA_EPS * Phi^-1(P_L))."""
    vs1 = np.asarray(vs1, dtype=float)
    num = ((TH1 * vs1) ** TH2
           - TH4 * np.log(mw)
           - TH5 * np.log(sigma_eff_kpa)
           + TH6 * np.asarray(fc, dtype=float)
           + SIGMA_EPS * norm.ppf(pl))
    return np.exp(num / TH3)


def dwf(mw):
    """Eq. (17): duration weighting factor, valid 5.5 < Mw < 9.0."""
    return 15.0 * np.asarray(mw, dtype=float) ** (-1.342)


def fs_liq(vs1, csr, mw, sigma_eff_kpa, fc=0.0):
    """Eq. (16): deterministic FS at the paper's recommended P_L = 15% boundary."""
    return crr(vs1, mw, sigma_eff_kpa, fc, pl=0.15) / np.asarray(csr, dtype=float)


def self_test(verbose=True):
    """Unit tests anchored to statements in the source papers. Returns dict."""
    out = {}

    # T1: DWF at the reference magnitude 7.5 is ~1.0.
    out["T1_dwf75"] = float(dwf(7.5))
    assert abs(out["T1_dwf75"] - 1.0) < 0.01, out["T1_dwf75"]

    # T2: erratum-corrected Eq.(14) exactly inverts Eq.(13).
    # T2a, algebraic identity via the margin (no cdf/ppf saturation):
    rng = np.random.default_rng(42)
    vs1 = rng.uniform(110, 350, 200)
    csr0 = rng.uniform(0.05, 0.7, 200)
    mw = rng.uniform(5.9, 9.0, 200)
    sig = rng.uniform(20, 180, 200)
    fc = rng.uniform(0, 35, 200)
    s = margin(vs1, csr0, mw, sig, fc)
    num = (TH1 * vs1) ** TH2 - TH4 * np.log(mw) - TH5 * np.log(sig) + TH6 * fc + SIGMA_EPS * s
    csr_back = np.exp(num / TH3)
    out["T2a_identity_maxrelerr"] = float(np.max(np.abs(csr_back / csr0 - 1.0)))
    assert out["T2a_identity_maxrelerr"] < 1e-9, out["T2a_identity_maxrelerr"]
    # T2b, cdf/ppf round trip where P_L is numerically representable (|s| <= 6):
    keep = np.abs(s) <= 6.0
    pl = p_liq(vs1[keep], csr0[keep], mw[keep], sig[keep], fc[keep])
    csr_back2 = crr(vs1[keep], mw[keep], sig[keep], fc[keep], pl=pl)
    out["T2b_roundtrip_n"] = int(keep.sum())
    out["T2b_roundtrip_maxrelerr"] = float(np.max(np.abs(csr_back2 / csr0[keep] - 1.0)))
    assert out["T2b_roundtrip_n"] > 50
    assert out["T2b_roundtrip_maxrelerr"] < 1e-6, out["T2b_roundtrip_maxrelerr"]

    # T3: P_L = 0.5 exactly when CSR = CRR(0.5).
    c50 = crr(180.0, 7.0, 60.0, 10.0, pl=0.5)
    out["T3_pl_at_crr50"] = float(p_liq(180.0, c50, 7.0, 60.0, 10.0))
    assert abs(out["T3_pl_at_crr50"] - 0.5) < 1e-12

    # T4: mechanics-sign monotonicity of the margin.
    base = margin(180.0, 0.25, 7.0, 60.0, 10.0)
    assert margin(200.0, 0.25, 7.0, 60.0, 10.0) < base      # stiffer -> safer
    assert margin(180.0, 0.30, 7.0, 60.0, 10.0) > base      # more demand -> riskier
    assert margin(180.0, 0.25, 7.5, 60.0, 10.0) > base      # longer duration -> riskier
    out["T4_monotonic"] = True

    # T5: catalog-mean sanity (GSP-334 Fig. 4: Vs1 174+-37, CSR 0.26+-0.14;
    # KEA2013 mean sigma'_v0 = 60.2 kPa). Liquefaction-rich catalog: P_L high.
    out["T5_pl_mean_case"] = float(p_liq(174.0, 0.26, 7.5, 60.2, 0.0))
    assert 0.5 < out["T5_pl_mean_case"] < 0.99
    # A clearly safe case sits far on the other side.
    out["T5_pl_safe_case"] = float(p_liq(240.0, 0.10, 6.5, 60.2, 0.0))
    assert out["T5_pl_safe_case"] < 0.05

    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    print("vs_kayen2013 self-test:")
    self_test()
    print("ALL PASS")
