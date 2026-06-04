"""CPT profile processing for liquefaction triggering + manifestation (zero-shot).

Robertson (2010) unit weight, Robertson (2009) soil-behaviour-type index Ic (iterative),
FC-from-Ic per Boulanger & Idriss (2014), effective stress with a water table, BI2014-CPT
CRR over the profile -> per-depth FS -> (a) critical-layer FS, (b) Liquefaction Potential
Index LPI (Iwasaki et al. 1978) as the triggering->surface-manifestation bridge.
Units: stress kPa, depth m, gamma kN/m^3. No coefficient is fitted to labels.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from . import crr_boulanger_idriss as BI
from . import csr as CSR

PA = 101.325
GW = 9.81


def unit_weight_robertson(qt, fs, pa=PA, gw=GW):
    qt = np.asarray(qt, float)
    fs = np.asarray(fs, float)
    Rf = np.clip(np.where(qt > 0, fs / np.maximum(qt, 1e-6) * 100.0, 1.0), 0.1, 12.0)
    qtpa = np.clip(qt / pa, 1e-3, None)
    gamma = gw * (0.27 * np.log10(Rf) + 0.36 * np.log10(qtpa) + 1.236)
    return np.clip(gamma, 14.0, 22.0)


def vertical_stresses(depth, gamma, gwt, gw=GW):
    depth = np.asarray(depth, float)
    gamma = np.asarray(gamma, float)
    sv = np.concatenate([[0.0], cumulative_trapezoid(gamma, depth)])
    u = gw * np.maximum(depth - gwt, 0.0)
    sev = np.maximum(sv - u, 0.1)
    return sv, u, sev


def ic_robertson(qt, fs, sv, sev, pa=PA, n_iter=8):
    qt = np.asarray(qt, float)
    qnet = np.maximum(qt - np.asarray(sv, float), 1.0)
    F = np.clip(np.asarray(fs, float) / qnet * 100.0, 0.1, 10.0)
    sev = np.asarray(sev, float)
    n = np.full_like(qt, 0.5)
    Ic = np.full_like(qt, 2.0)
    for _ in range(n_iter):
        Q = np.clip((qnet / pa) * (pa / sev) ** n, 1e-3, None)
        Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
        n = np.clip(0.381 * Ic + 0.05 * (sev / pa) - 0.15, None, 1.0)
    return Ic, F, n


def fc_from_ic(Ic, cfc=0.0):
    return np.clip(80.0 * (np.asarray(Ic, float) + cfc) - 137.0, 0.0, 100.0)


def process_case(depth, qc, fs, gwt, Mw, pga, ic_liq=2.6, zmin=1.0, zmax=20.0, sev_min=10.0):
    """Return critical-layer FS and LPI summary for one CPT case (zero-shot)."""
    depth = np.asarray(depth, float)
    qc = np.asarray(qc, float)
    fs = np.asarray(fs, float)
    gamma = unit_weight_robertson(qc, fs)
    sv, u, sev = vertical_stresses(depth, gamma, gwt)
    Ic, F, n = ic_robertson(qc, fs, sv, sev)
    qc1Ncs = BI.normalize_cpt(qc, sev, fc_from_ic(Ic))["qc1Ncs"]
    rd = CSR.stress_reduction_rd(depth, Mw)
    csr = CSR.csr_seed_idriss(pga, sv, sev, rd)
    crr = BI.crr_insitu(Mw, sev, qc1Ncs, mode="cpt")
    with np.errstate(divide="ignore", invalid="ignore"):
        fs_liq = np.where(csr > 0, crr / csr, np.inf)

    liq = (Ic < ic_liq) & (depth > gwt) & np.isfinite(fs_liq)
    # LPI (Iwasaki et al. 1978): integral of F(z)*w(z), F=1-FS if FS<1 (liquefiable), w=10-0.5z
    Fz = np.where(liq & (fs_liq < 1.0), 1.0 - fs_liq, 0.0)
    Fz = np.clip(Fz, 0.0, 1.0)
    wz = np.clip(10.0 - 0.5 * depth, 0.0, None) * (depth <= 20.0)
    LPI = float(trapezoid(Fz * wz, depth))

    mask = liq & (depth > max(gwt, zmin)) & (depth <= zmax) & (sev >= sev_min)
    if not np.any(mask):
        return {"crit_FS": np.inf, "crit_depth": np.nan, "qc1Ncs_crit": np.nan,
                "Ic_crit": np.nan, "sev_crit": np.nan, "n_liq_layers": 0, "LPI": LPI}
    sub = np.where(mask)[0]
    j = sub[int(np.argmin(fs_liq[sub]))]
    return {"crit_FS": float(fs_liq[j]), "crit_depth": float(depth[j]),
            "qc1Ncs_crit": float(qc1Ncs[j]), "Ic_crit": float(Ic[j]),
            "sev_crit": float(sev[j]), "n_liq_layers": int(mask.sum()), "LPI": LPI}
