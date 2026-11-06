"""Water v5 (pre-registration 2026-11-06): soil-evaporation scale, grid choice, weekly score, closure rule."""
import numpy as np
import pandas as pd
import pytest

import phase37_water_v5 as F
from phase29_water_v2 import dual_kc_balance


def _inputs(n=30):
    et0 = np.full(n, 4.0)
    rain = np.zeros(n); rain[0] = 20.0
    lai = np.linspace(0.2, 3.0, n)
    kcb = 0.15 + (1.1 - 0.15) * (1 - np.exp(-0.7 * lai))
    return et0, rain, kcb, lai, np.full(n, 150.0)


def test_ke_scale_default_unchanged_and_scales_evaporation():
    et0, rain, kcb, lai, taw = _inputs()
    a = dual_kc_balance(et0, rain, kcb, lai, taw, 0.55, 25.0)
    b = dual_kc_balance(et0, rain, kcb, lai, taw, 0.55, 25.0, ke_scale=1.0)
    c = dual_kc_balance(et0, rain, kcb, lai, taw, 0.55, 25.0, ke_scale=0.5)
    assert np.allclose(a["et"], b["et"])
    assert c["e"][0] == pytest.approx(0.5 * a["e"][0])
    assert c["t"][0] == pytest.approx(a["t"][0])
    assert c["e"].sum() < a["e"].sum()


def test_choose_prefers_lowest_then_closest_to_v2():
    scores = {(1.0, 0.70): 5.0, (0.8, 0.50): 4.0, (0.6, 0.40): 4.0, (0.5, 0.30): 4.5}
    assert F.choose(scores) == (0.8, 0.50)


def test_weekly_rmse():
    d = pd.date_range("2022-04-01", periods=14)
    model = pd.Series(np.r_[np.full(7, 3.0), np.full(7, 2.0)], index=d)
    tower = pd.Series(np.r_[np.full(7, 2.0), np.full(7, 2.0)], index=d)
    valid = pd.Series(True, index=d)
    valid.iloc[7:10] = False                     # second week: 4 valid days -> dropped
    r, n, logr = F.weekly_scores(model, tower, valid, d[0])
    assert n == 1 and r == pytest.approx(7.0) and logr == pytest.approx(np.log(1.5))


def test_closure_rule():
    assert F.deciding(0.62) and F.deciding(1.05) and not F.deciding(1.10)
