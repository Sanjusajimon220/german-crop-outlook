import numpy as np
import pandas as pd

from phase21_crop_stage import doy, fit_weight, loso_scores, near, predict


def test_doy_relative_to_the_harvest_year():
    assert doy(np.datetime64("2023-05-20"), 2023) == 140
    assert doy(np.datetime64("2022-10-15"), 2023) == -78                        # autumn before the harvest year


def test_near_returns_indices_within_radius():
    xy = np.array([[0.0, 0.0], [3000.0, 0.0], [6000.0, 0.0]])
    assert near(xy, (0.0, 4000.0), 5000).tolist() == [0, 1]


def test_fit_weight_finds_satellite_share_and_offset():
    rng = np.random.default_rng(0)
    cal = rng.normal(150, 3, 60)
    sat = cal + rng.normal(0, 6, 60)                                           # satellite carries extra field information
    obs = 0.6 * sat + 0.4 * cal + 4.0                                          # truth: 60 % satellite, observed 4 days later
    w, k = fit_weight(cal, sat, obs)
    assert abs(w - 0.6) < 0.051 and abs(k - 4.0) < 0.5
    w0, _ = fit_weight(cal, cal + rng.normal(0, 6, 60), cal + 2)               # satellite is pure noise -> calendar only
    assert w0 <= 0.1


def test_predict_blends_calendar_and_satellite():
    assert predict(np.array([150.0]), np.array([160.0]), 0.5, 2.0)[0] == 157.0


def test_loso_scores_hold_out_each_season():
    rows = []
    for y in (2021, 2022, 2023, 2024):
        for i in range(10):
            cal = 150 + i
            rows.append(dict(season=y, obs=cal + 3, cal=cal, sat=cal + 5, hrvpp=cal + 10))   # obs = cal + 3 exactly
    s = loso_scores(pd.DataFrame(rows))
    assert s["calendar"]["rmse"] < 1e-9 and s["ours"]["rmse"] < 1e-9 and s["hrvpp"]["rmse"] < 1e-9   # constant offsets learned
    assert s["calendar"]["n"] == 40
