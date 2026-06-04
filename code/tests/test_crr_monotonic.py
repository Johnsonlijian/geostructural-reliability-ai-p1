import numpy as np
from geoliq.mechanics import crr_boulanger_idriss as bi


def test_crr75_increases_with_resistance():
    assert bi.crr75_spt(10) < bi.crr75_spt(20) < bi.crr75_spt(30)
    assert bi.crr75_cpt(80) < bi.crr75_cpt(120) < bi.crr75_cpt(160)


def test_msf_unity_at_M7_5_and_monotone():
    # MSF ~= 1 at M=7.5 by construction; BI2014 constants give 0.9999935 (not exactly 1).
    assert abs(float(bi.msf(7.5, 20.0, "spt")) - 1.0) < 1e-3
    assert float(bi.msf(6.0, 20.0, "spt")) > 1.0
    assert float(bi.msf(8.0, 20.0, "spt")) < 1.0


def test_ksigma_caps_and_decreases_with_stress():
    assert bi.k_sigma(200.0, 20.0, "spt") < bi.k_sigma(50.0, 20.0, "spt")
    assert float(bi.k_sigma(10.0, 20.0, "spt")) <= 1.1 + 1e-9


def test_normalization_cn_cap_and_fines_increase():
    r = bi.normalize_spt(15.0, 100.0, 15.0)
    assert float(r["CN"]) <= 1.7 + 1e-9
    assert float(r["N1_60cs"]) >= float(r["N1_60"])
    rc = bi.normalize_cpt(8000.0, 100.0, 15.0)
    assert float(rc["CN"]) <= 1.7 + 1e-9
    assert float(rc["qc1Ncs"]) >= float(rc["qc1N"])


def test_fs_positive():
    crr = bi.crr_insitu(7.0, 80.0, 18.0, "spt")
    assert float(bi.fs_liquefaction(crr, 0.2)) > 0
