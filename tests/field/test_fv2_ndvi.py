import numpy as np

import fv2


def test_ndvi_normal_values():
    assert abs(fv2.ndvi(np.array([0.05]), np.array([0.40]))[0] - 0.7778) < 1e-3


def test_ndvi_invalid_when_bands_near_zero():
    v = fv2.ndvi(np.array([0.002, -0.004]), np.array([0.003, 0.0035]))
    assert np.isnan(v).all()


def test_ndvi_never_outside_minus_one_to_one():
    red = np.array([0.02, 0.5, -0.01])
    nir = np.array([0.30, -0.6, 0.30])
    v = fv2.ndvi(red, nir)
    assert np.all(np.isnan(v) | (np.abs(v) <= 1))
