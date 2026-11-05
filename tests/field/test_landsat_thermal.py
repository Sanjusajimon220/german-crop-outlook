import numpy as np

from phase31_landsat_thermal import kelvin, window


def test_window_is_april_to_september_of_the_season_year():
    assert window(2023) == ("2023-04-01", "2023-09-30")


def test_kelvin_uses_collection2_scale_and_keeps_nodata_as_nan():
    k = kelvin(np.array([0, 44000], np.uint16))
    assert np.isnan(k[0]) and abs(k[1] - (44000 * 0.00341802 + 149.0)) < 1e-9
