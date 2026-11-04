import numpy as np

from phase26_sl2p_weekly import domain_codes, line_weekly, pair_stats


def test_domain_codes_match_the_set_lookup_of_phase4_sl2p():
    lo, hi = np.zeros(8), np.ones(8)
    x = np.array([[0.05] * 8, [0.95] * 8, [np.nan] + [0.5] * 7])     # cells 1..., 10..., invalid
    grid = {tuple([1] * 8)}
    ok = np.isin(domain_codes(x, lo, hi), domain_codes(np.array([[0.05] * 8]), lo, hi))
    assert ok.tolist() == [True, False, False]
    assert ok.tolist() == [bool(np.isfinite(r).all()) and tuple((np.floor(10 * r) + 1).astype(int)) in grid
                           for r in x]


def test_line_weekly_measured_line_and_last_value():
    days = np.array([0, 40, 50])                                    # clear days
    vals = np.array([[1.0], [3.0], [np.nan]])                       # third date cloudy for this pixel
    weeks = np.array([-5, 2, 20, 45, 70])
    out, measured = line_weekly(days, vals, weeks)
    assert np.isnan(out[0, 0])                                     # before the first clear image
    assert out[1, 0] == 1.0 and measured[1, 0]                     # within 3 days: measured
    assert out[2, 0] == 2.0 and not measured[2, 0]                 # 40-day gap: straight line, whatever the gap
    assert out[3, 0] == 3.0 and not measured[3, 0]                 # after the last clear image: last value
    assert out[4, 0] == 3.0


def test_pair_stats_on_known_differences():
    a, b = np.array([1.0, 2.0, 3.0, np.nan]), np.array([1.5, 2.0, 2.5, 4.0])
    s = pair_stats(a, b)
    assert s["n"] == 3 and abs(s["mean_diff"]) < 1e-12 and abs(s["rmsd"] - np.sqrt(0.5 / 3)) < 1e-12
