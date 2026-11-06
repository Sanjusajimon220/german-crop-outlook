"""Test-site field chain (water v4 pre-registration 2026-11-06): outline, interior, season bounds."""
import numpy as np
import pandas as pd

import phase33_testsite_field as T


def _stack(field_mask, n=12, seed=0):
    """NDVI dates x rows x cols: one curve inside the field, a different (shifted) curve outside, small noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n)
    inside, outside = 0.2 + 0.6 * np.sin(t), 0.2 + 0.6 * np.sin(t + 1.5) ** 2
    a = np.where(field_mask[None], inside[:, None, None], outside[:, None, None])
    return a + rng.normal(0, 0.01, a.shape)


def test_outline_keeps_connected_field_around_tower():
    m = np.zeros((40, 40), bool)
    m[5:25, 8:30] = True
    m[30:38, 30:38] = True                      # same crop, but not connected to the tower field
    nd = _stack(m)
    out = T.outline(nd, (15, 15), radius_px=60)
    assert out[5:25, 8:30].all()
    assert not out[30:38, 30:38].any()
    assert out.sum() == 20 * 22


def test_outline_needs_enough_common_dates():
    m = np.zeros((20, 20), bool)
    m[2:18, 2:18] = True
    nd = _stack(m, n=12)
    nd[5:, :, 10:] = np.nan                     # right half: only 5 clear dates
    out = T.outline(nd, (8, 4), min_dates=8)
    assert out[2:18, 2:10].all() and not out[:, 10:].any()


def test_outline_respects_radius():
    m = np.ones((50, 50), bool)
    out = T.outline(_stack(m), (25, 25), radius_px=10)
    rr, cc = np.nonzero(out)
    assert np.sqrt((rr - 25) ** 2 + (cc - 25) ** 2).max() <= 10


def test_interior_distance():
    m = np.zeros((60, 60), bool)
    m[10:50, 10:50] = True
    inner = T.interior(m, 10)                   # 10 px = 100 m
    assert inner[20:40, 20:40].all() and inner.sum() == 20 * 20


def test_season_bounds():
    d = pd.date_range("2021-03-01", "2021-09-30")
    lai = np.interp(np.arange(len(d)), [0, 40, 110, 150, 214], [0.2, 0.6, 5.0, 0.3, 0.0])
    s, p, e = T.season_bounds(d, lai)
    assert s == d[np.argmax(lai >= 0.5)]
    assert p == d[110]
    assert e == d[110 + np.argmax(lai[110:] < 0.5)]


def test_season_bounds_winter_crop_starts_1_march():
    d = pd.date_range("2021-03-01", "2021-09-30")
    lai = np.interp(np.arange(len(d)), [0, 90, 140, 214], [1.0, 6.0, 0.2, 0.0])
    s, p, e = T.season_bounds(d, lai)
    assert s == d[0] and p == d[90]


def test_season_bounds_after_sowing_ignores_cover_crop():
    d = pd.date_range("2019-03-01", "2019-09-30")
    lai = np.interp(np.arange(len(d)), [0, 36, 60, 110, 130, 175, 214], [1.4, 2.46, 0.3, 0.3, 2.38, 2.2, 2.2])
    s, p, e = T.season_bounds(d, lai, after=pd.Timestamp("2019-04-20"))
    assert p == d[130]
    low = np.flatnonzero(lai[:130] < 0.5)                  # emergence: the day after the last bare day before the peak
    assert s == d[low[-1] + 1] and s > pd.Timestamp("2019-05-15")
