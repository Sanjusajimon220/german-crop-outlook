import numpy as np
import pandas as pd

from phase28_water_job3 import fao56_et, kc_curve, scores


def test_kc_curve_follows_the_fao56_shape():
    dates = pd.date_range("2022-01-01", "2022-08-31", freq="D")
    st = {"emergence": pd.Timestamp("2021-10-25"), "heading": pd.Timestamp("2022-05-20"), "maturity": pd.Timestamp("2022-07-25")}
    kc = pd.Series(kc_curve(dates, st, kc_end=0.3), index=dates)
    assert kc["2022-02-01"] == 0.4                                  # winter: Kc_ini
    assert abs(kc["2022-05-20"] - 1.15) < 1e-9                      # heading: Kc_mid
    half = st["heading"] + (st["maturity"] - st["heading"]) / 2
    assert abs(kc[half.normalize()] - 1.15) < 0.01                  # mid until half of grain filling
    assert abs(kc["2022-07-25"] - 0.3) < 1e-9 and kc["2022-08-20"] == 0.3
    assert 0.4 < kc["2022-04-01"] < 1.15                            # development: in between


def test_fao56_et_without_stress_is_et0_times_kc_and_stress_limits_it():
    et0, kc = np.full(10, 4.0), np.full(10, 1.0)
    wet = fao56_et(et0, rain=np.full(10, 10.0), kc=kc, taw=np.full(10, 150.0), p=0.55)
    assert np.allclose(wet, 4.0)                                    # no stress: ET = ET0 x Kc
    dry = fao56_et(np.full(60, 5.0), rain=np.zeros(60), kc=np.ones(60), taw=np.full(60, 100.0), p=0.55)
    assert dry[0] == 5.0 and dry[-1] < 1.0                          # bucket empties: Ks < 1
    dep = np.cumsum(dry)
    assert dep[-1] <= 100.0 + 1e-9                                  # never more than the available water


def test_scores_daily_and_season():
    s = scores(np.array([2.0, 3.0, 4.0]), np.array([2.0, 2.5, 4.5]))
    assert abs(s["ratio"] - 9.0 / 9.0) < 1e-12 and abs(s["rmse"] - np.sqrt(0.5 / 3)) < 1e-12
