"""Held-out water test (pre-registration 2026-11-06): soil ruling, tower rain, pass rule."""
import numpy as np
import pandas as pd
import pytest

import phase38_water_heldout as H


def test_silt_loam_soil():
    paw, tew = H.silt_loam_soil()
    assert len(paw) == 20 and paw == pytest.approx(np.full(20, 14.0))     # (0.29 - 0.15) x 100 mm per 10 cm
    assert tew == pytest.approx(1000 * (0.29 - 0.5 * 0.15) * 0.10)       # FAO-56 eq. 73


def test_daily_rain():
    t = pd.date_range("2022-06-01 00:30", periods=96, freq="30min")
    hh = pd.DataFrame({"TIMESTAMP_END": t.strftime("%Y%m%d%H%M").astype(np.int64), "P_F": np.r_[np.full(48, 0.1), np.zeros(48)]})
    r = H.daily_rain(hh)
    assert r.iloc[0] == pytest.approx(4.8) and r.iloc[1] == pytest.approx(0.0)


def test_pass_rule():
    assert H.passes(rmse_v5=4.5, rmse_v2=5.0, log_v5=np.log(1.08), better=6, seasons=10)
    assert not H.passes(rmse_v5=4.6, rmse_v2=5.0, log_v5=0.0, better=10, seasons=10)          # only -8 %
    assert not H.passes(rmse_v5=4.0, rmse_v2=5.0, log_v5=np.log(1.12), better=10, seasons=10)  # bias 12 %
    assert not H.passes(rmse_v5=4.0, rmse_v2=5.0, log_v5=0.0, better=5, seasons=10)            # 50 % of seasons
