import numpy as np
import pandas as pd

import phase11_fusion_test as F


def _w(days, tmin, tas, pr):
    idx = pd.date_range("2024-01-01", periods=days)
    return pd.DataFrame({"tasmin": tmin, "tas": tas, "pr": pr}, index=idx)


def test_frost_snow_and_heavy_rain_flag_a_radar_date():
    w = _w(10, [-5, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 4, 4, 0.0, 4, 4, 4, 4, 4, 4], [0, 0, 0, 3, 0, 0, 6, 6, 0, 0])
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-05", "2024-01-08", "2024-01-10"])
    flags = F.radar_flags(dates, w)
    # 2 Jan: frost the day before; 5 Jan: snow (3 mm at 0 C) within 3 days; 8 Jan: 12 mm in the previous 2 days; 10 Jan: clean
    assert flags.tolist() == [True, True, True, False]


def test_nearest_unflagged_radar_date():
    s1 = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-06"])
    ok = np.array([True, False, True])
    assert F.nearest_ok(s1, pd.Timestamp("2024-01-03"), ok) == 0            # 3 Jan flagged: 1 Jan (2 days) is nearest unflagged
    assert F.nearest_ok(s1, pd.Timestamp("2024-01-10"), ok) == -1           # nothing unflagged within 3 days


def test_measured_snow_depth_takes_priority_over_the_weather_proxy():
    w = _w(6, [2] * 6, [4, 4, 4, 0.0, 4, 4], [0, 0, 0, 3, 0, 0])     # the proxy would call 5 Jan snowy (3 mm at 0 C on 4 Jan)
    snow_cm = pd.Series([0, 0, 0, 0, 0, 5], index=w.index)            # the station measured no snow until 6 Jan
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-06"])
    assert F.radar_flags(dates, w, snow_cm).tolist() == [False, True]
    gaps = pd.Series([0, 0, 0, None, None, None], index=w.index, dtype=float)   # no station value -> proxy
    assert F.radar_flags(dates, w, gaps).tolist() == [True, True]                  # proxy: 4 Jan snowfall lies within 3 days of both
