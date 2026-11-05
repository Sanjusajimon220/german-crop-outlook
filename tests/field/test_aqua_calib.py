import numpy as np
import pandas as pd

from phase0_modis_aqua_calib import apply, fit, pair, scores


def test_pair_matches_the_nearest_aqua_composite_within_8_days():
    t = pd.DataFrame({"date": pd.to_datetime(["2010-05-09", "2010-05-25"]), "district": ["A", "A"], "terra": [0.6, 0.7]})
    a = pd.DataFrame({"date": pd.to_datetime(["2010-05-17", "2010-06-30"]), "district": ["A", "A"], "aqua": [0.65, 0.8]})
    p = pair(t, a, max_days=8)
    assert len(p) == 2 and p.aqua.tolist() == [0.65, 0.65]          # both Terra dates 8 days from the 17 May Aqua


def test_fit_and_apply_recover_a_known_line_per_month():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.2, 0.9, 400)
    month = np.repeat([5, 6], 200)
    terra = np.where(month == 5, 0.02 + 0.95 * a, -0.01 + 1.05 * a)
    df = pd.DataFrame({"aqua": a, "terra": terra, "month": month})
    c = fit(df, by_month=True)
    out = apply(c, df.assign(terra=np.nan))
    assert np.allclose(out, terra, atol=1e-9)
    one = fit(df, by_month=False)
    assert set(one) == {0}                                             # one line for all months


def test_scores():
    s = scores(np.array([0.5, 0.6, 0.7]), np.array([0.5, 0.62, 0.68]))
    assert abs(s["rmse"] - np.sqrt((0 + 0.0004 + 0.0004) / 3)) < 1e-12 and abs(s["r"] - np.corrcoef([0.5, 0.6, 0.7], [0.5, 0.62, 0.68])[0, 1]) < 1e-12
