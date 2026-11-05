import numpy as np

from phase29_water_v2 import dual_kc_balance
from phase30_water_v3 import ensemble_balance, radar_wetness, weigh


def test_ensemble_balance_equals_v2_for_each_version():
    rng = np.random.default_rng(0)
    n = 60
    et0, rain = rng.uniform(1, 5, n), np.where(rng.random(n) < 0.3, rng.uniform(2, 15, n), 0.0)
    kcb, lai = np.linspace(0.15, 1.1, n), np.linspace(0, 5, n)
    taw = np.linspace(40, 180, n)
    v = dict(taw_f=np.array([0.7, 1.3]), kcb_f=np.array([0.9, 1.1]), dr0=np.array([0.0, 0.3]), tew_f=np.array([1.0, 0.8]))
    out = ensemble_balance(et0, rain, kcb, lai, taw, p=0.55, tew=22.0, versions=v)
    for k in range(2):
        ref = dual_kc_balance(et0, rain, np.minimum(kcb * v["kcb_f"][k], 1.2), lai, taw * v["taw_f"][k], 0.55,
                              22.0 * v["tew_f"][k], dr0=v["dr0"][k] * taw[0] * v["taw_f"][k])
        assert np.allclose(out["et"][k], ref["et"]) and np.allclose(out["dr"][k], ref["dr"])


def test_radar_wetness_change_detection_is_0_to_1_per_season():
    vv = np.array([-16.0, -14.0, -12.0, -10.0, -8.0, np.nan])
    w = radar_wetness(vv)
    assert np.nanmin(w) == 0.0 and np.nanmax(w) == 1.0 and np.isnan(w[-1]) and w[2] > w[1]


def test_weigh_prefers_versions_close_to_the_observation_and_keeps_50_effective():
    model = np.tile(np.linspace(0, 1, 200)[:, None], (1, 3))        # 200 versions x 3 dates
    obs = np.array([0.8, 0.8, 0.8])
    w = weigh(model, obs, err=0.25)
    assert abs(w.sum() - 1) < 1e-12 and w[160] > w[20] and 1 / (w ** 2).sum() >= 50
    assert np.allclose(weigh(model[:, :0], obs[:0], err=0.25), 1 / 200)   # no observation: equal weights


def test_ensemble_balance_takes_a_kcb_per_version():
    n = 30
    et0, rain, lai = np.full(n, 4.0), np.zeros(n), np.full(n, 3.0)
    kcb2 = np.array([np.full(n, 0.6), np.full(n, 1.0)])               # Kcb_mid varied: whole rows given, not scaled
    v = dict(taw_f=np.ones(2), kcb_f=np.ones(2), dr0=np.zeros(2), tew_f=np.ones(2))
    out = ensemble_balance(et0, rain, kcb2, lai, np.full(n, 150.0), p=0.55, tew=22.0, versions=v)
    ref = dual_kc_balance(et0, rain, kcb2[1], lai, np.full(n, 150.0), 0.55, 22.0)
    assert np.allclose(out["et"][1], ref["et"]) and out["t"][0, 0] < out["t"][1, 0]


def test_weighted_quantile_follows_the_weights():
    from phase30_water_v3 import wquantile
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])              # 3 versions x 2 weeks
    assert np.allclose(wquantile(x, np.array([0.0, 1.0, 0.0]), 0.5), [2.0, 20.0])
    q = wquantile(x, np.full(3, 1 / 3), [0.1, 0.9])
    assert q.shape == (2, 2) and np.all(q[0] < q[1])


def test_weekly_sums_use_valid_tower_days_and_need_5_of_7():
    from phase30_water_v3 import weekly_sums
    days = np.arange(14)
    valid = np.ones(14, bool); valid[[1, 2, 3]] = False               # week 1 has 4 valid days -> dropped
    model = np.ones((2, 14)); tower = np.full(14, 2.0)
    m, t, keep = weekly_sums(model, tower, valid, days // 7)
    assert keep.tolist() == [False, True] and np.allclose(t[keep], 14.0) and np.allclose(m[:, keep], 7.0)
