import numpy as np

from phase29_water_v2 import dual_kc_balance, et0_pm, kcb_from_lai, u2_from_u10


def test_et0_matches_the_fao56_worked_example_18():
    # FAO-56 Example 18 (Brussels, 50 deg 48' N, 100 m, 6 July): Tmax 21.5, Tmin 12.3, RH 63-84 % (mean 73.5),
    # u2 2.078 m/s, Rs 22.07 MJ/m2/day -> ET0 3.9 mm/day (with RHmax / RHmin; RHmean gives slightly less)
    et0 = et0_pm(tmin=np.array([12.3]), tmax=np.array([21.5]), rh=np.array([73.5]), rs=np.array([22.07]),
                 u2=np.array([2.078]), elev=100.0, lat=50.8, doy=np.array([187]))
    assert abs(et0[0] - 3.9) < 0.25


def test_wind_height_conversion_fao56_eq47():
    assert abs(u2_from_u10(np.array([10 / 3.6]))[0] - 2.078) < 0.01


def test_kcb_from_lai_runs_from_bare_to_full_cover():
    k = kcb_from_lai(np.array([0.0, 3.0, 8.0]), kcb_mid=1.10)
    assert abs(k[0] - 0.15) < 1e-12 and k[0] < k[1] < k[2] < 1.10 and k[2] > 1.09


def test_dual_kc_bare_wet_soil_evaporates_then_dries_and_crop_stress_appears():
    n = 40
    out = dual_kc_balance(et0=np.full(n, 4.0), rain=np.r_[20.0, np.zeros(n - 1)], kcb=np.full(n, 0.15), lai=np.zeros(n),
                          taw=np.full(n, 100.0), p=0.55, tew=22.0, rew=9.0)
    e = out["e"]
    assert e[1] > 3.5 and e[-1] < 0.3                                 # stage 1 near Kc_max x ET0, then the surface dries
    full = dual_kc_balance(et0=np.full(n, 5.0), rain=np.zeros(n), kcb=np.full(n, 1.10), lai=np.full(n, 5.0),
                           taw=np.full(n, 100.0), p=0.55, tew=22.0, rew=9.0)
    assert full["ks"][0] == 1.0 and full["ks"][-1] < 1.0               # root zone dries -> water stress
    assert np.all(full["dr"] <= 100.0 + 1e-9)                          # never more than the available water
    assert np.allclose(full["et"], full["t"] + full["e"])
