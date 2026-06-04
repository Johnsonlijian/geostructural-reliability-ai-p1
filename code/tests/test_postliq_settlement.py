import numpy as np
from geoliq.mechanics import postliq_settlement as ps


def test_no_strain_when_fs_ge_2():
    assert np.isclose(ps.volumetric_strain(2.0, 0.5), 0.0)
    assert np.isclose(ps.volumetric_strain(2.5, 0.5), 0.0)


def test_strain_increases_as_fs_drops():
    assert ps.volumetric_strain(0.5, 0.5) > ps.volumetric_strain(1.5, 0.5)


def test_denser_soil_less_strain():
    assert ps.volumetric_strain(0.5, 0.8) < ps.volumetric_strain(0.5, 0.3)


def test_settlement_sums_layers_nonneg():
    assert ps.postliq_settlement([0.5, 0.8], [0.4, 0.4], [1.0, 1.0]) >= 0
