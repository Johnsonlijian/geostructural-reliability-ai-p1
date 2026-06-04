import numpy as np
from geoliq.mechanics import effective_stress as es


def test_water_table_at_depth_zero_pore():
    sv = es.total_vertical_stress(5.0, 5.0, 18.0, 20.0)
    sev = es.effective_vertical_stress(5.0, 5.0, 18.0, 20.0)
    assert np.isclose(sv, 90.0)
    assert np.isclose(sev, sv)  # u = 0 at z = dw


def test_deeper_water_table_increases_effective_stress():
    z = 10.0
    sev_shallow = es.effective_vertical_stress(z, 3.0, 18.0, 20.0)
    sev_deep = es.effective_vertical_stress(z, 5.0, 18.0, 20.0)
    assert sev_deep > sev_shallow


def test_effective_le_total_and_positive():
    z = np.array([2.0, 6.0, 12.0])
    sv = es.total_vertical_stress(z, 3.0, 18.0, 20.0)
    sev = es.effective_vertical_stress(z, 3.0, 18.0, 20.0)
    assert np.all(sev <= sv + 1e-9)
    assert np.all(sev > 0)
