"""T4a — Seismic demand: stress reduction coefficient rd (Idriss 1999) and CSR (Seed-Idriss).

CSR = 0.65 * (a_max/g) * (sigma_v / sigma'_v) * rd
rd  = exp(alpha(z) + beta(z)*Mw),  z<=34 m ;  rd = 0.12*exp(0.22*Mw), z>34 m
"""
import numpy as np


def stress_reduction_rd(z, Mw):
    z = np.asarray(z, float)
    Mw = np.asarray(Mw, float)
    alpha = -1.012 - 1.126 * np.sin(z / 11.73 + 5.133)
    beta = 0.106 + 0.118 * np.sin(z / 11.28 + 5.142)
    rd_shallow = np.exp(alpha + beta * Mw)
    rd_deep = 0.12 * np.exp(0.22 * Mw)
    return np.where(z <= 34.0, rd_shallow, rd_deep)


def csr_seed_idriss(pga_g, sigma_v, sigma_eff_v, rd):
    pga_g = np.asarray(pga_g, float)
    sigma_v = np.asarray(sigma_v, float)
    sigma_eff_v = np.asarray(sigma_eff_v, float)
    rd = np.asarray(rd, float)
    return 0.65 * pga_g * (sigma_v / sigma_eff_v) * rd
