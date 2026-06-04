import numpy as np
from geoliq.mechanics import csr


def test_rd_near_surface_about_one():
    assert abs(float(csr.stress_reduction_rd(0.0, 7.5)) - 1.0) < 0.02


def test_rd_decreases_with_depth():
    rd = csr.stress_reduction_rd(np.array([1.0, 5.0, 10.0, 20.0]), 7.5)
    assert np.all(np.diff(rd) < 0)
    assert np.all((rd > 0) & (rd <= 1.05))


def test_csr_increases_with_pga_and_lower_eff_stress():
    base = float(csr.csr_seed_idriss(0.2, 100.0, 60.0, 0.95))
    assert float(csr.csr_seed_idriss(0.4, 100.0, 60.0, 0.95)) > base
    assert float(csr.csr_seed_idriss(0.2, 100.0, 40.0, 0.95)) > base
