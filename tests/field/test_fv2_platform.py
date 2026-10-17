import numpy as np

import fv2_platform as X


def test_alert_needs_two_weeks_beyond_uncertainty():
    med = [2.0] * 6
    unc = [0.4] * 6
    one = [2.0, 1.5, 2.0, 2.0, 2.0, 2.0]          # one week below by more than the uncertainty
    two = [2.0, 1.5, 1.5, 2.0, 2.0, 2.0]          # two weeks in a row
    assert X.alerts(one, unc, med) == [0, 0, 0, 0, 0, 0]
    assert X.alerts(two, unc, med) == [0, 0, 1, 0, 0, 0]


def test_alert_falling_while_median_rises():
    med = [1.0, 1.5, 2.0, 2.5]
    fld = [1.0, 1.0, 0.9, 0.6]                    # falls while the median rises
    unc = [0.3] * 4
    a = X.alerts(fld, unc, med)
    assert a[3] in (1, 2) and a[0] == 0


def test_alerts_ignore_missing_weeks():
    assert X.alerts([np.nan, 1.0, 1.0], [0.4] * 3, [2.0, 2.0, 2.0]) == [0, 0, 1]


def test_off_season_label():
    # winter crop: green before sowing (previous crop), drop to ~0.2, rise after sowing, harvest drop
    lai = [1.5, 1.2, 0.3, 0.2, 0.3, 0.8, 1.5, 3.0, 3.2, 1.0, 0.3, 0.2]
    m = X.season_mask(lai, "wheat")
    assert m[0] == 0 and m[1] == 0                # previous crop / before sowing
    assert m[7] == 1                              # in season
    assert m[-1] == 0                             # after harvest


def test_edge_fields_flagged_partial():
    assert X.partial(0.95) is True
    assert X.partial(0.995) is False


def test_series_matches_pixel_mean():
    vals = np.array([[1.0, 3.0, 5.0, np.nan]])
    w = np.array([1.0, 1.0, 0.5, 1.0])
    inv = np.array([0, 0, 1, 1])
    out = X.field_mean(vals, w, inv, n_fields=2)
    assert np.allclose(out, [[2.0, 5.0]])
