import numpy as np
import pandas as pd

from fv2_benchmark_export import area_test_rows, item2_points, item3_boxes, item4_series


def test_area_test_rows_counts_wins_over_line_for_long_gaps():
    s = pd.DataFrame({"target": ["lai"] * 4, "area": ["a", "a", "b", "b"], "year": [2021, 2022, 2021, 2022],
                      "gap": [">30 d"] * 4, "line": [0.8, 0.5, 0.7, 0.6], "design": [0.6, 0.6, 0.5, 0.5]})
    out = area_test_rows(s, "lai")
    assert out["won"] == 3 and out["of"] == 4
    assert out["areas"] == [{"area": "a", "line": 0.65, "design": 0.6}, {"area": "b", "line": 0.65, "design": 0.5}]


def test_item2_points_and_totals():
    t = pd.DataFrame({"date": ["p|2021|x", "p|2022|y", "p|2022|z"], "n": [40, 50, 60], "rho_ours": [0.9, 0.7, 0.8],
                      "rho_ppi": [0.8, 0.75, 0.8]})
    out = item2_points(t)
    assert out["won"] == 1 and out["of"] == 3                           # a tie is not a win
    assert out["points"][0] == {"season": 2021, "n": 40, "ours": 0.9, "ppi": 0.8}


def test_item3_boxes_quartiles_per_crop_and_event():
    t = pd.DataFrame({"crop": ["wheat"] * 5, "event": ["peak"] * 5, "diff_days": [-10, 0, 5, 10, 20]})
    b = item3_boxes(t)
    assert b == [{"crop": "wheat", "event": "peak", "q1": 0.0, "median": 5.0, "q3": 10.0, "n": 5}]


def test_item4_series_mean_per_area_and_date_with_before_and_after():
    a = pd.DataFrame({"mode": "all", "area": "pilot", "date": ["2021-05-10"] * 2, "ours": [1.0, 2.0], "cgls": [2.0, 2.0]})
    b = a.assign(ours=[1.5, 2.5])
    s = item4_series(a, b)
    assert s["pilot"] == [["2021-05-10", 2.0, 1.5, 2.0]]               # date, CGLS, ours before, ours after
    assert np.isclose(s["pilot"][0][2], 1.5)
