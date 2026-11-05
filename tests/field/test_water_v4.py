"""Water v4 (thermal, pre-registration 2026-11-06): overpass weather, solar ratio, screen, interior pixels, TSEB, weights."""
import numpy as np
import pytest

import phase34_water_v4 as V


def test_air_temp_sine():
    sr = 5.0
    assert V.air_temp_at(10.0, 20.0, sr, sr) == pytest.approx(10.0)
    assert V.air_temp_at(10.0, 20.0, 14.0, sr) == pytest.approx(20.0)
    mid = V.air_temp_at(10.0, 20.0, 9.5, sr)
    assert 10.0 < mid < 20.0 and mid > 15.0               # sine rise is fast early


def test_rs_instant_integrates_to_daily():
    n = 15.0
    hours = np.linspace(12 - n / 2, 12 + n / 2, 2001)
    w = V.rs_instant(20.0, hours, n)                        # W/m2
    assert np.trapezoid(w, hours) * 3600 / 1e6 == pytest.approx(20.0, rel=1e-3)
    assert V.rs_instant(20.0, 12.0, n) == pytest.approx(w.max(), rel=1e-6)


def test_daily_et_solar_ratio():
    # LE 300 W/m2 at Rs 600 W/m2 -> evaporative share of sunlight 0.5; daily Rs 20 MJ -> 10 MJ -> 10e6 / 2.45e6 mm
    assert V.daily_et(300.0, 600.0, 20.0) == pytest.approx(10e6 / 2.45e6)


def test_screen():
    lst = np.array([300.0, 300.0, 300.0, 290.0, 300.0])
    qa_pixel = np.array([64, 64 | 8, 64, 64, 0])            # clear, cloud bit 3, clear, clear, not clear
    st_qa = np.array([1.0, 1.0, 4.0, 1.0, 1.0])
    ok = V.screen(lst, qa_pixel, st_qa, t_air=296.0)        # 290 < 296 - 5 -> cold, dropped
    assert ok.tolist() == [True, False, False, False, False]


def test_interior_landsat_pixels():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True                               # 600 m square on the 10 m grid
    # Landsat 30 m pixel centres at 10 m pixel (row, col) positions
    rows = np.array([50, 25, 35])
    cols = np.array([50, 50, 50])
    ok = V.interior_points(mask, rows, cols, min_m=100.0)
    assert ok.tolist() == [True, False, True]


def test_tseb_et_plausible():
    out = V.tseb(tr_k=300.0, ta_k=298.0, lai=3.0, h_c=0.8, u=3.0, ea_mb=15.0, p_mb=1000.0, rs=700.0, sza=40.0)
    assert out["flag"] < 255
    assert 100.0 < out["le"] < 700.0
    hot = V.tseb(tr_k=312.0, ta_k=298.0, lai=3.0, h_c=0.8, u=3.0, ea_mb=15.0, p_mb=1000.0, rs=700.0, sza=40.0)
    assert hot["le"] < out["le"]                            # a hotter canopy evaporates less


def test_obs_error():
    assert V.obs_error(np.array([1.0, 4.0])).tolist() == pytest.approx([0.5, 1.2])


def test_active_seasons_skip_dropped(monkeypatch):
    monkeypatch.setattr(V, "_info", lambda a, y: {"dropped": True} if (a, y) == ("dekli", 2022) else {"crop": "wheat"})
    assert ("dekli", 2022) not in V.active(V.SEPARATE + V.JUDGED)
    assert V.active(V.JUDGED) == V.JUDGED


def test_usable_handles_empty_season():
    import pandas as pd
    u = V.usable(pd.DataFrame())
    assert len(u) == 0 and {"date", "et_thermal"} <= set(u.columns)


def test_active_drops_seasons_without_interior_landsat_pixels(monkeypatch):
    monkeypatch.setattr(V, "_info", lambda a, y: {"crop": "potato"})
    monkeypatch.setattr(V, "_interior_px", lambda a, y: 0 if (a, y) == ("degeb", 2020) else 50)
    assert V.active(V.INDICATIVE) == [("dekli", 2018)]
