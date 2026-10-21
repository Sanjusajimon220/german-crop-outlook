import numpy as np
from affine import Affine

from phase16_hrvpp_items import (decode_ppi, decode_yydoy, grid_index, item2_score, nearest_date, season_timing)


def test_decode_ppi_scales_and_masks_nodata():
    out = decode_ppi(np.array([0, 15000, -32768], np.int16))
    assert out[0] == 0.0 and abs(out[1] - 1.5) < 1e-9 and np.isnan(out[2])


def test_decode_yydoy():
    d = decode_yydoy(np.array([18030, 21095, 0]))
    assert str(d[0]) == "2018-01-30" and str(d[1]) == "2021-04-05" and np.isnat(d[2])


def test_grid_index_nearest_cell_same_crs():
    t = Affine(10, 0, 1000, 0, -10, 2000)                     # 10 m cells, top-left (1000, 2000)
    r, c = grid_index(np.array([1005.0, 1019.9, 1001.0]), np.array([1995.0, 1985.0, 1900.0]), t, (5, 5), 25832, 25832)
    assert r.tolist() == [0, 1, -1] and c.tolist() == [0, 1, -1]  # third point outside the 5 x 5 raster


def test_nearest_date_within_tolerance():
    dates = np.array(["2023-05-01", "2023-05-11", "2023-05-21"], dtype="datetime64[D]")
    assert nearest_date(dates, np.datetime64("2023-05-14"), 5) == 1
    assert nearest_date(dates, np.datetime64("2023-06-10"), 5) == -1


def test_season_timing_half_amplitude_rule():
    lai = np.array([0.5, 0.6, 1.0, 2.5, 4.5, 4.0, 2.0, 0.6, 0.5])
    mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0], bool)          # in-season weeks 1..7
    s, p, e = season_timing(lai, mask)                          # min 0.6, max 4.5, threshold 2.55
    assert (s, p, e) == (4, 4, 5)


def test_item2_score_counts_strict_wins_on_dates_with_enough_fields():
    rng = np.random.default_rng(0)
    truth = rng.random(40)
    rows = [("d1", truth, truth + 0.01 * rng.random(40), rng.random(40)),   # ours ~ truth, ppi random -> ours wins
            ("d2", truth, rng.random(40), truth),                            # ppi = truth -> ppi wins
            ("d3", truth[:20], truth[:20], rng.random(20))]                  # only 20 fields -> not counted
    res = item2_score(rows, min_fields=30)
    assert res["dates"] == 2 and res["ours_wins"] == 1
