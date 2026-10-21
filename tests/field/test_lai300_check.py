import numpy as np
import pandas as pd
from affine import Affine

from phase14_lai300_check import aggregate, nearest_week, pixel_index


def test_pixel_index_maps_centres_to_300m_cells():
    t = Affine(0.5, 0, 6.0, 0, -0.5, 51.0)                       # 0.5 deg cells, top-left (6.0, 51.0)
    r, c = pixel_index(np.array([6.1, 6.6, 6.9]), np.array([50.9, 50.4, 50.6]), t)
    assert r.tolist() == [0, 1, 0] and c.tolist() == [0, 1, 1]


def test_aggregate_applies_inside_and_coverage_rules():
    # cell (0,0): 10 of 10 expected pixels, values 1..10; cell (0,1): only 4 of 10 have values -> dropped (< 50 %)
    r = np.array([0] * 10 + [0] * 4); c = np.array([0] * 10 + [1] * 4)
    v = np.r_[np.arange(1, 11, dtype=float), np.ones(4)]
    expected = {(0, 0): 10.0, (0, 1): 10.0}
    inside = {(0, 0): 0.95, (0, 1): 1.0}
    out = aggregate(r, c, v, expected, inside)
    assert list(out.index) == [(0, 0)]
    assert out.loc[(0, 0), "ours"] == 5.5 and out.loc[(0, 0), "share"] == 1.0


def test_aggregate_drops_cells_mostly_outside_the_area():
    r = np.zeros(10, int); c = np.zeros(10, int); v = np.ones(10)
    assert aggregate(r, c, v, {(0, 0): 10.0}, {(0, 0): 0.89}).empty


def test_nearest_week_within_three_days_only():
    weeks = pd.to_datetime(["2023-06-05", "2023-06-12", "2023-06-19"]).values.astype("datetime64[D]")
    assert nearest_week(weeks, np.datetime64("2023-06-10")) == 1
    assert nearest_week(weeks, np.datetime64("2023-06-30")) == -1
