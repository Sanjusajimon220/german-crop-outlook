import numpy as np

import fv2_weekly as W

days = np.array([0, 20, 60])                       # clear dates (days since season start)
vals = np.array([1.0, 2.0, 4.0])
sd = np.array([0.1, 0.2, 0.3])


def test_measured_week_keeps_s2_value():
    v, u, src, gap = W.assign_week(days, vals, sd, 22)
    assert (v, u, src) == (2.0, 0.2, 1)


def test_gap_classes():
    v, u, src, gap = W.assign_week(days, vals, sd, 10)
    assert src == 2 and gap == 20 and abs(v - 1.5) < 1e-9
    v, u, src, gap = W.assign_week(days, vals, sd, 40)
    assert src == 3 and gap == 40


def test_after_last_clear_image():
    v, u, src, gap = W.assign_week(days, vals, sd, 80)
    assert src == 4


def test_no_clear_value_gives_missing():
    v, u, src, gap = W.assign_week(np.array([], int), np.array([]), np.array([]), 10)
    assert np.isnan(v) and src == 0
    v, u, src, gap = W.assign_week(days, vals, sd, -10)
    assert np.isnan(v) and src == 0


def test_weekly_matches_assign_week():
    """Integration (real data, pilot 2024): the vectorised weekly series picks the same source, gap and (for measured
    and line weeks) the same value as the pure assign_week rule, on 300 random pixels."""
    import os
    from areas import out_dir
    weeks = W.mondays(2024)
    z = np.load(os.path.join(out_dir("pilot", 2024), W.OUTFILE))
    S = W.season_on_weeks("pilot", 2024, "lai", weeks)
    s2 = S.s2_rows
    rng = np.random.default_rng(1)
    for j in rng.choice(S.Y.shape[1], 300, replace=False):
        ok = np.isfinite(S.Y[s2, j])
        days, vals = S.days[s2][ok], S.Y[s2, j][ok]
        for w, wd in enumerate(weeks):
            v, _, src, gap = W.assign_week(days, vals, np.zeros(len(vals)), S.days[S.dates.get_loc(wd)])
            assert z["lai_src"][w, j] == src, (j, str(wd.date()))
            if src in (W.SRC_LINE, W.SRC_RADAR):
                assert z["lai_gap"][w, j] == gap, (j, str(wd.date()))
            if src in (W.SRC_MEASURED, W.SRC_LINE):
                assert abs(z["lai"][w, j] - max(v, 0)) < 1e-4, (j, str(wd.date()))
