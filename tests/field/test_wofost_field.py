import datetime as dt

import numpy as np

from phase22_wofost_field import enkf_update, lhs, soil_site, weekly_sample


def test_soil_site_from_bk50_millimetres():
    soil, site = soil_site(fk_mm=330.0, nfk_mm=165.0, root_dm=11.0, water_f=1.0)
    assert abs(soil["SMFCF"] - 0.30) < 1e-9                    # 330 mm in 1,100 mm of soil
    assert abs(soil["SMW"] - 0.15) < 1e-9                      # field capacity minus plant-available water
    assert soil["RDMSOL"] == 110.0                             # cm
    wet, _ = soil_site(330.0, 165.0, 11.0, water_f=1.2)
    assert wet["SMW"] < soil["SMW"]                            # more plant-available water -> lower wilting point


def test_lhs_spreads_every_setting_over_its_range():
    m = lhs(100, seed=1)
    assert len(m) == 100
    for col, lo, hi in (("tsum1_f", 0.88, 1.12), ("tsum2_f", 0.88, 1.12), ("sla_f", 0.8, 1.2), ("water_f", 0.8, 1.2)):
        v = m[col].values
        assert v.min() >= lo and v.max() <= hi
        assert np.histogram(v, bins=10, range=(lo, hi))[0].min() == 10     # one value in every tenth of the range per 10
    assert m["sow_shift"].between(-7, 7).all()


def test_weekly_sample_reads_mondays_and_zero_outside_the_run():
    out = [{"day": dt.date(2024, 4, 1) + dt.timedelta(days=i), "LAI": float(i)} for i in range(10)]
    weeks = np.array(["2024-03-25", "2024-04-01", "2024-04-08", "2024-04-15"], dtype="datetime64[D]")
    assert weekly_sample(out, weeks).tolist() == [0.0, 0.0, 7.0, 0.0]


def test_enkf_update_moves_ensemble_towards_the_observation():
    rng = np.random.default_rng(0)
    ens = rng.normal(2.0, 0.5, 200)
    upd = enkf_update(ens, y=3.0, r=0.5 ** 2, rng=rng)
    assert 2.3 < upd.mean() < 2.7                              # about halfway: equal model and observation spread
    assert upd.std() < ens.std()
    assert (upd >= 0).all()
