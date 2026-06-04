"""T4b — Boulanger & Idriss (2014, UCD/CGM-14/01) CRR for CPT/SPT, with MSF and Ksigma.

Zero-shot: NO coefficient here is fitted to liquefaction labels. Verify all constants
against the original report before reporting.
"""
import numpy as np

PA = 101.325  # kPa, atmospheric pressure


def normalize_cpt(qc_kpa, sigma_eff_v, fc_pct, pa=PA, n_iter=50):
    """CPT overburden normalization + fines correction -> qc1Ncs (iterative, CN<=1.7)."""
    qc = np.asarray(qc_kpa, float)
    sev = np.asarray(sigma_eff_v, float)
    fc = np.asarray(fc_pct, float)
    qc1Ncs = np.full(np.broadcast(qc, sev, fc).shape, 100.0)
    CN = qc1N = m = None
    for _ in range(n_iter):
        m = 1.338 - 0.249 * np.clip(qc1Ncs, 21.0, 254.0) ** 0.264
        CN = np.minimum((pa / sev) ** m, 1.7)
        qc1N = CN * (qc / pa)
        dqc1N = (11.9 + qc1N / 14.6) * np.exp(1.63 - 9.7 / (fc + 2.0) - (15.7 / (fc + 2.0)) ** 2)
        qc1Ncs = qc1N + dqc1N
    return {"qc1N": qc1N, "qc1Ncs": qc1Ncs, "CN": CN, "m": m}


def normalize_spt(N60, sigma_eff_v, fc_pct, pa=PA, n_iter=50):
    """SPT overburden normalization + fines correction -> N1_60cs (iterative, CN<=1.7)."""
    N = np.asarray(N60, float)
    sev = np.asarray(sigma_eff_v, float)
    fc = np.asarray(fc_pct, float)
    N1_60cs = np.full(np.broadcast(N, sev, fc).shape, 20.0)
    CN = N1_60 = m = None
    for _ in range(n_iter):
        m = 0.784 - 0.0768 * np.sqrt(np.clip(N1_60cs, 0.0, 46.0))
        CN = np.minimum((pa / sev) ** m, 1.7)
        N1_60 = CN * N
        dN = np.exp(1.63 + 9.7 / (fc + 0.01) - (15.7 / (fc + 0.01)) ** 2)
        N1_60cs = N1_60 + dN
    return {"N1_60": N1_60, "N1_60cs": N1_60cs, "CN": CN, "m": m}


def crr75_cpt(qc1Ncs):
    # BI2014: qc1Ncs > ~211 is effectively non-liquefiable; clip to keep CRR finite.
    q = np.clip(np.asarray(qc1Ncs, float), None, 211.0)
    return np.exp(q / 113.0 + (q / 1000.0) ** 2 - (q / 140.0) ** 3 + (q / 137.0) ** 4 - 2.80)


def crr75_spt(N1_60cs):
    n = np.asarray(N1_60cs, float)
    return np.exp(n / 14.1 + (n / 126.0) ** 2 - (n / 23.6) ** 3 + (n / 25.4) ** 4 - 2.80)


def msf(Mw, resistance_cs, mode):
    Mw = np.asarray(Mw, float)
    r = np.asarray(resistance_cs, float)
    if mode == "cpt":
        msf_max = np.minimum(1.09 + (r / 180.0) ** 3, 2.2)
    elif mode == "spt":
        msf_max = np.minimum(1.09 + (r / 31.5) ** 2, 2.2)
    else:
        raise ValueError("mode must be 'cpt' or 'spt'")
    return 1.0 + (msf_max - 1.0) * (8.64 * np.exp(-Mw / 4.0) - 1.325)


def k_sigma(sigma_eff_v, resistance_cs, mode, pa=PA):
    sev = np.asarray(sigma_eff_v, float)
    r = np.asarray(resistance_cs, float)
    if mode == "cpt":
        Csig = np.minimum(1.0 / (37.3 - 8.27 * np.clip(r, None, 211.0) ** 0.264), 0.3)
    elif mode == "spt":
        Csig = np.minimum(1.0 / (18.9 - 2.55 * np.sqrt(np.clip(r, 0.0, 37.0))), 0.3)
    else:
        raise ValueError("mode must be 'cpt' or 'spt'")
    return np.minimum(1.0 - Csig * np.log(sev / pa), 1.1)


def crr_insitu(Mw, sigma_eff_v, resistance_cs, mode, pa=PA):
    base = crr75_cpt(resistance_cs) if mode == "cpt" else crr75_spt(resistance_cs)
    return base * msf(Mw, resistance_cs, mode) * k_sigma(sigma_eff_v, resistance_cs, mode, pa)


def fs_liquefaction(crr, csr):
    return np.asarray(crr, float) / np.asarray(csr, float)
