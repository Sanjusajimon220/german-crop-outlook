import numpy as np

import fv2_weekly as W

days = np.array([0, 20, 60])                       # clear dates (days since season start)
vals = np.array([1.0, 2.0, 4.0])
sd = np.array([0.1, 0.2, 0.3])


def test_measured_week_keeps_s2_value():
    v, u, src, gap = W.assign_week(days, vals, sd, 22)
    assert (v, u, src) == (2.0, 0.2, 1)


def test_gap_classes():
    v, u, src, gap = W.assign_week(days, vals, sd, 10)
    assert src == 2 and gap == 20 and abs(v - 1.5) < 1e-9
    v, u, src, gap = W.assign_week(days, vals, sd, 40)
    assert src == 3 and gap == 40


def test_after_last_clear_image():
    v, u, src, gap = W.assign_week(days, vals, sd, 80)
    assert src == 4


def test_no_clear_value_gives_missing():
    v, u, src, gap = W.assign_week(np.array([], int), np.array([]), np.array([]), 10)
    assert np.isnan(v) and src == 0
    v, u, src, gap = W.assign_week(days, vals, sd, -10)
    assert np.isnan(v) and src == 0
