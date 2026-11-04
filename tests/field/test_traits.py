import numpy as np

from phase27_traits import SRC_AFTER, SRC_LINE, SRC_MEASURED, gap_errors, trait_weekly, week_uncertainty


def test_trait_weekly_sources_and_gaps():
    days = np.array([0, 40, 50])
    vals = np.array([[10.0], [30.0], [np.nan]])
    weeks = np.array([-5, 2, 20, 45, 70])
    v, src, gap = trait_weekly(days, vals, weeks)
    assert np.isnan(v[0, 0]) and src[0, 0] == 0                     # before the first clear image
    assert v[1, 0] == 10.0 and src[1, 0] == SRC_MEASURED and gap[1, 0] == 0
    assert v[2, 0] == 20.0 and src[2, 0] == SRC_LINE and gap[2, 0] == 40     # gap length between the two clear values
    assert v[3, 0] == 30.0 and src[3, 0] == SRC_AFTER and gap[3, 0] == 5     # days since the last clear value
    assert v[4, 0] == 30.0 and src[4, 0] == SRC_AFTER and gap[4, 0] == 30


def test_week_uncertainty_uses_network_sd_when_measured_and_the_table_otherwise():
    unc = {"line_le15": 1.0, "line_16_30": 2.0, "line_gt30": 3.0, "after_le30": 4.0, "after_gt30": 5.0}
    src = np.array([[SRC_MEASURED, SRC_LINE, SRC_LINE, SRC_LINE, SRC_AFTER, SRC_AFTER, 0]])
    gap = np.array([[0, 10, 20, 45, 30, 31, 0]])
    sd = np.array([[0.5] * 7])
    u = week_uncertainty(src, gap, sd, unc)
    assert u[0, :6].tolist() == [0.5, 1.0, 2.0, 3.0, 4.0, 5.0] and np.isnan(u[0, 6])


def test_gap_errors_line_and_held_value():
    days = np.array([0, 10, 20, 60])
    vals = np.array([[0.0], [5.0], [20.0], [20.0]])                 # hide day 10 and day 20 in turn
    rows = gap_errors(days, vals, hide=np.array([False, True, False, False]))
    by = {(r["kind"], r["cls"]): r for r in rows}
    assert by[("line", "le15")]["n"] == 0 and by[("line", "16_30")]["n"] == 1
    assert abs(by[("line", "16_30")]["sse"] - (10.0 - 5.0) ** 2) < 1e-9        # line 0 -> 20 gives 10 at day 10
    assert by[("after", "le30")]["n"] == 1 and abs(by[("after", "le30")]["sse"] - 25.0) < 1e-9   # held 0 vs 5
