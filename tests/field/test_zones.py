import numpy as np

from phase18_zones import merge_small_patches, relative_growth, season_sum, tertile_zones, zone_order_score


def test_season_sum_needs_most_in_season_weeks():
    lai = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, np.nan], [0.5, np.nan]])     # weeks x pixels
    mask = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], bool)                        # in-season weeks 1-2 for both
    out = season_sum(lai, mask, min_cover=0.8)
    assert out[0] == 5.0 and np.isnan(out[1])                                       # pixel 1 has 0 of 2 in-season weeks


def test_relative_growth_divides_by_weighted_field_mean():
    s = np.array([2.0, 4.0, 3.0, np.nan])
    field = np.array([0, 0, 1, 1])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    rel = relative_growth(s, field, w)
    assert np.allclose(rel[:3], [2 / 3, 4 / 3, 1.0]) and np.isnan(rel[3])


def test_tertile_zones_per_field():
    v = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.0, 1.0, 1.0])
    field = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    z = tertile_zones(v, field)
    assert z[:6].tolist() == [1, 1, 2, 2, 3, 3]
    assert set(z[6:].tolist()) <= {1, 2, 3}


def test_merge_small_patches_absorbs_specks_into_neighbours():
    g = np.full((6, 6), 2, np.int8)
    g[2, 2] = 3                                                                       # a single-pixel speck
    g[:, 5] = 1                                                                       # a large enough strip
    out = merge_small_patches(g, min_pixels=3)
    assert out[2, 2] == 2 and (out[:, 5] == 1).all()


def test_zone_order_score():
    zones = np.array([1, 1, 2, 2, 3, 3])
    assert zone_order_score(zones, np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.2])) == 1.0
    assert zone_order_score(zones, np.array([1.0, 1.0, 0.8, 0.9, 1.1, 1.2])) == 0.5   # weak and middle swapped
    assert np.isnan(zone_order_score(np.array([1, 1, 2, 2]), np.array([1.0, 1.0, 1.0, 1.0])))  # needs all 3 zones
