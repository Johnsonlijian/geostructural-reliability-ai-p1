import numpy as np
from geoliq.mechanics import gb50011 as gb


def test_clay_content_floor():
    a = gb.gb50011_ncr(10, 0.95, 8.0, 2.0, 1.0)
    b = gb.gb50011_ncr(10, 0.95, 8.0, 2.0, 3.0)
    assert np.isclose(a, b)


def test_ncr_decreases_with_water_table_depth():
    shallow = gb.gb50011_ncr(10, 0.95, 8.0, 1.0, 5.0)
    deep = gb.gb50011_ncr(10, 0.95, 8.0, 5.0, 5.0)
    assert deep < shallow


def test_liquefied_flag():
    ncr = float(gb.gb50011_ncr(10, 0.95, 8.0, 2.0, 5.0))
    assert bool(gb.gb50011_liquefied(ncr - 1.0, 10, 0.95, 8.0, 2.0, 5.0)) is True
    assert bool(gb.gb50011_liquefied(ncr + 1.0, 10, 0.95, 8.0, 2.0, 5.0)) is False
