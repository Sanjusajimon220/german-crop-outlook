"""Thermal stress alerts (pre-registration 2026-11-06): scores, simple rule, stress index."""
import numpy as np
import pandas as pd
import pytest

import phase36_thermal_stress as S


def test_scores():
    alert = np.array([True, True, False, False, True, False])
    truth = np.array([True, False, False, True, True, False])
    s = S.scores(alert, truth)
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["false_alarm"] == pytest.approx(1 / 3)
    assert s["balanced_accuracy"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert s["n"] == 6 and s["stressed"] == 3


def test_scores_one_class_is_nan():
    s = S.scores(np.array([True, False]), np.array([False, False]))
    assert np.isnan(s["hit_rate"]) and np.isnan(s["balanced_accuracy"])


def test_simple_rule():
    d = pd.date_range("2022-06-01", periods=12)
    rain = pd.Series([10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0], index=d, dtype=float)
    et0 = pd.Series(3.5, index=d)
    r = S.simple_rule(rain, et0)
    assert not r.iloc[9]                    # window days 0-9 holds the 10 mm
    assert r.iloc[10] and r.iloc[11]        # days 1-10: 3 mm; days 2-11: 3 mm
    et0.iloc[11] = 2.0
    assert not S.simple_rule(rain, et0).iloc[11]


def test_stress_index():
    # Kc_pot = Kcb(LAI) + 0.05; index = ET / (Kc_pot x ET0)
    from phase29_water_v2 import kcb_from_lai
    k = float(kcb_from_lai(np.array([3.0]), 1.10)[0]) + 0.05
    assert S.stress_index(2.0, 3.0, 4.0, "wheat") == pytest.approx(2.0 / (k * 4.0))
