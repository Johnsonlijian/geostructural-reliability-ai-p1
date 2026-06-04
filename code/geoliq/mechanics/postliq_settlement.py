"""T5 — Post-liquefaction reconsolidation settlement (Ishihara-Yoshimine framework).

WARNING: VERIFIED = False. The volumetric-strain magnitudes are a transparent,
shape-correct PLACEHOLDER (monotone in FS and Dr), NOT authoritative. Replace with
Yoshimine et al. (2006) / Idriss & Boulanger (2008) coefficients before any reporting.
"""
import numpy as np

VERIFIED = False


def dr_from_n1_60cs(N1_60cs):
    """Relative density from SPT (Idriss-Boulanger): Dr = sqrt(N1_60cs/46)."""
    return np.sqrt(np.clip(np.asarray(N1_60cs, float), 0.0, None) / 46.0)


def volumetric_strain(fs_liq, Dr):
    """PLACEHOLDER eps_v(FS, Dr): 0 at FS>=2, grows as FS->0, lower for denser soil."""
    fs = np.asarray(fs_liq, float)
    Dr = np.asarray(Dr, float)
    trig = np.clip(2.0 - fs, 0.0, None) / 2.0
    return np.clip(0.06 * np.exp(-3.0 * Dr) * trig, 0.0, 0.10)


def postliq_settlement(fs_liq_layers, Dr_layers, dz_layers):
    ev = volumetric_strain(fs_liq_layers, Dr_layers)
    return float(np.sum(ev * np.asarray(dz_layers, float)))
