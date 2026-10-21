import numpy as np

from phase19_zone_checks import fill_sinks, flow_accumulation, slope_deg, twi, zone_contrast


def test_slope_of_a_plane():
    y, x = np.mgrid[0:20, 0:20]
    dem = x * 10.0                                            # 10 m cells, rise 10 m per cell -> 45 degrees
    s = slope_deg(dem, 10.0)
    assert abs(s[10, 10] - 45.0) < 0.5


def test_fill_sinks_raises_a_pit_to_its_spill_level():
    dem = np.full((5, 5), 10.0); dem[2, 2] = 5.0              # a pit in a flat plateau
    dem[0, :] = 9.0                                           # lower outlet edge
    f = fill_sinks(dem)
    assert 10.0 <= f[2, 2] <= 10.01                           # filled to its rim (+ the tiny drainage tilt)


def test_flow_accumulation_collects_downhill():
    y, x = np.mgrid[0:6, 0:6]
    dem = (10 - x).astype(float)                              # slopes down to the east
    acc = flow_accumulation(dem)
    assert acc[3, 5] > acc[3, 0]                              # more upslope cells at the lower (east) edge
    assert acc[3, 0] == 1                                     # a ridge cell drains only itself


def test_twi_higher_in_a_valley_than_on_a_ridge():
    y, x = np.mgrid[0:30, 0:30]
    dem = np.abs(x - 15).astype(float) * 2.0 + y * 0.5         # V-shaped valley along column 15, sloping south
    t = twi(dem, 10.0)
    assert t[20, 15] > t[20, 2]


def test_zone_contrast_weak_minus_strong_per_field():
    zone = np.array([1, 1, 3, 3, 1, 3])
    var = np.array([50.0, 60.0, 100.0, 110.0, 80.0, 70.0])
    field = np.array([0, 0, 0, 0, 1, 1])
    out = zone_contrast(zone, var, field)
    assert out.loc[0, "weak_minus_strong"] == -50.0 and out.loc[1, "weak_minus_strong"] == 10.0
