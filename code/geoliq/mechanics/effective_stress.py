"""T3 — vertical total / pore / effective stress with a water table at depth dw.

Units: stress kPa, unit weight kN/m^3, depth m. gamma_w = 9.81 kN/m^3.
"""
import numpy as np

GAMMA_W = 9.81  # kN/m^3


def total_vertical_stress(z, dw, gamma_moist, gamma_sat):
    z = np.asarray(z, float)
    dw = np.asarray(dw, float)
    above = np.asarray(gamma_moist, float) * np.minimum(z, dw)
    below = np.asarray(gamma_sat, float) * np.maximum(z - dw, 0.0)
    return above + below


def pore_pressure(z, dw, gamma_w=GAMMA_W):
    z = np.asarray(z, float)
    dw = np.asarray(dw, float)
    return gamma_w * np.maximum(z - dw, 0.0)


def effective_vertical_stress(z, dw, gamma_moist, gamma_sat, gamma_w=GAMMA_W):
    return total_vertical_stress(z, dw, gamma_moist, gamma_sat) - pore_pressure(z, dw, gamma_w)
