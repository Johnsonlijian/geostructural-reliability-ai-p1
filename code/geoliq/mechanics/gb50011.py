"""T4c — GB 50011-2010 §4.3.4 SPT-based liquefaction discrimination.

Ncr = N0 * beta * (ln(0.6*ds + 1.5) - 0.1*dw) * sqrt(3/rho_c),  ds <= 20 m
Liquefied if measured N63.5 < Ncr.  VERIFY N0/beta tables against the code text.
"""
import numpy as np

# Reference blow count N0 by design PGA (g). GB 50011-2010 Table 4.3.4. VERIFY.
N0_BY_PGA_G = {0.10: 7, 0.15: 10, 0.20: 12, 0.30: 16, 0.40: 19}
# Adjustment factor beta by design earthquake group (1/2/3).
BETA_BY_GROUP = {1: 0.80, 2: 0.95, 3: 1.05}


def gb50011_ncr(N0, beta, ds, dw, rho_c):
    ds = np.asarray(ds, float)
    dw = np.asarray(dw, float)
    rho_c = np.maximum(np.asarray(rho_c, float), 3.0)  # clay content floor = 3%
    return N0 * beta * (np.log(0.6 * ds + 1.5) - 0.1 * dw) * np.sqrt(3.0 / rho_c)


def gb50011_liquefied(N_measured, N0, beta, ds, dw, rho_c):
    return np.asarray(N_measured, float) < gb50011_ncr(N0, beta, ds, dw, rho_c)
