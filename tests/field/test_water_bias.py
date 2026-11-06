"""FAO-56 bias study (pre-registration 2026-11-06): tower ET0, wind height, daily tower data, stages, attribution."""
import numpy as np
import pandas as pd
import pytest

import phase35_water_bias as B
from phase29_water_v2 import et0_pm, u2_from_u10


def test_et0_ea_matches_et0_pm():
    tmin, tmax, rh, rs, u2 = np.array([8.0, 12.0]), np.array([20.0, 27.0]), np.array([70.0, 55.0]), \
        np.array([15.0, 24.0]), np.array([2.0, 3.5])
    es_ = lambda x: 0.6108 * np.exp(17.27 * x / (x + 237.3))
    ea = rh / 100 * (es_(tmax) + es_(tmin)) / 2
    p = 101.3 * ((293 - 0.0065 * 100) / 293) ** 5.26
    a = B.et0_ea(tmin, tmax, ea, rs, u2, p, 100.0, 50.9, np.array([150, 200]))
    b = et0_pm(tmin, tmax, rh, rs, u2, 100.0, 50.9, np.array([150, 200]))
    assert a == pytest.approx(b, rel=1e-9)


def test_u2_from_height():
    assert B.u2_from_height(5.0, 10.0) == pytest.approx(float(u2_from_u10(5.0)))
    assert B.u2_from_height(4.0, 2.0) == pytest.approx(4.0, rel=2e-3)


def test_tower_daily_aggregation():
    t = pd.date_range("2022-06-01 00:30", periods=96, freq="30min")
    ta = np.r_[np.linspace(10, 20, 48), np.linspace(12, 24, 48)]
    hh = pd.DataFrame({"TIMESTAMP_END": t.strftime("%Y%m%d%H%M").astype(np.int64), "TA_F": ta, "VPD_F": 5.0,
                       "SW_IN_F": 200.0, "WS_F": 3.0, "PA_F": 100.0})
    d = B.tower_daily(hh)
    assert list(d.index) == list(pd.to_datetime(["2022-06-01", "2022-06-02"]))
    assert d["tmax"].iloc[0] == pytest.approx(20.0) and d["tmin"].iloc[1] == pytest.approx(12.0)
    assert d["rs"].iloc[0] == pytest.approx(200 * 0.0864)
    es_ = lambda x: 0.6108 * np.exp(17.27 * x / (x + 237.3))
    assert d["ea"].iloc[0] == pytest.approx(np.mean(es_(ta[:48]) - 0.5))
    assert d["ws"].iloc[0] == pytest.approx(3.0) and d["pa"].iloc[0] == pytest.approx(100.0)


def test_stages():
    d = pd.date_range("2022-04-01", periods=6)
    lai = np.array([0.5, 2.0, 3.5, 4.0, 3.0, 0.4])
    peak = d[3]
    assert B.stages(d, lai, peak).tolist() == ["lai<1", "lai1-3", "lai>=3", "lai>=3", "after_peak", "after_peak"]


def test_attribution_adds_up():
    et_m, et_t = np.array([20.0, 30.0]), np.array([16.0, 25.0])
    e0_m, e0_t = np.array([25.0, 32.0]), np.array([22.0, 30.0])
    a = B.attribution(et_m, et_t, e0_m, e0_t)
    assert a["total"] == pytest.approx(a["et0"] + a["kc"])
    assert a["total"] == pytest.approx(np.mean(np.log(et_m / et_t)))
