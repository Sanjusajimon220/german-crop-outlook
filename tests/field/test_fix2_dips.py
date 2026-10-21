import numpy as np
import pandas as pd

from phase15_fix2_dips import candidates, compare_groups, dip_flags


def test_candidates_need_drop_below_both_neighbours():
    # pixel 0: 3.0, 1.0, 3.2 with sd 0.2 -> dip; pixel 1: 3.0, 2.6, 3.0 -> not (0.4 < 3 x 0.2); pixel 2: dip at the first value
    Y = np.array([[3.0, 3.0, 0.5], [1.0, 2.6, 3.0], [3.2, 3.0, 3.1]])
    sd = np.full(Y.shape, 0.2)
    t, p, a, b = candidates(Y, sd)
    assert list(zip(t, p)) == [(1, 0)] and a.tolist() == [0] and b.tolist() == [2]


def test_flag_only_when_radar_does_not_drop():
    y_a, y_i, y_b, sd = np.array([3.0, 3.0]), np.array([1.0, 1.0]), np.array([3.2, 3.2]), np.array([0.2, 0.2])
    radar = np.array([2.9, 1.2])                                 # pixel 1: radar shows the drop -> real, not flagged
    assert dip_flags(y_a, y_i, y_b, sd, radar).tolist() == [True, False]


def test_compare_groups_uses_same_pixels_and_min_count():
    s2 = np.r_[np.full(12, 1.0), np.full(12, 3.0), np.full(5, 0.0)]
    ls = np.r_[np.full(12, 3.0), np.full(12, 3.1), np.full(5, 3.0)]
    flag = np.r_[np.ones(12, bool), np.zeros(12, bool), np.ones(5, bool)]
    field = np.r_[np.zeros(24, int), np.ones(5, int)]           # field 1 has only 5 flagged pixels -> dropped
    out = compare_groups(s2, ls, flag, field, min_px=10)
    assert len(out) == 2
    assert out.set_index("flagged").loc[True, "absdiff"] == 2.0
    assert round(out.set_index("flagged").loc[False, "absdiff"], 3) == 0.1
