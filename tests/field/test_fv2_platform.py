import numpy as np
import pytest

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


def test_export_redone_unless_complete(tmp_path, monkeypatch):
    # a shutdown can leave series.json without the pixel and image files; only the '.complete' marker means done
    monkeypatch.setattr(X, "DOCS", str(tmp_path))
    d = tmp_path / "nowhere_2024"
    d.mkdir()
    (d / "series.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        X.export("nowhere", 2024)
    (d / ".complete").write_text("")
    assert X.export("nowhere", 2024) is None


def test_off_season_winter_frost_dip_is_not_sowing():
    # review 2026-10-20: stubble in October, crop 0.2-0.3 from November, a frost-affected dip to 0.08 in February
    lai = [0.1, 0.1, 0.1, 0.1, 0.2] + [0.3] * 13 + [0.08] + [0.4, 0.8, 1.5, 2.5, 3.5, 4.0, 4.0, 3.5, 2.5, 1.0, 0.4] + [0.3] * 18
    m = X.season_mask(lai, "wheat")
    assert m[5] == 1 and m[17] == 1               # November to January is the wheat, not 'before sowing'
    assert m[25] == 1


def test_off_season_spring_crop_cover_crop_is_not_the_crop():
    # review 2026-10-20: bare after the October harvest, winter cover crop, then maize from June
    lai = [0.02] * 4 + [0.6, 0.8, 1.0, 1.0, 0.9, 0.8] + [0.8] * 12 + [0.6, 0.4, 0.2, 0.15, 0.15, 0.2, 0.3] + \
          [0.6, 1.2, 2.0, 3.0, 3.5, 3.5, 3.0, 2.5] + [2.0] * 9
    m = X.season_mask(lai, "maize")
    assert m[6] == 0 and m[15] == 0               # cover crop in winter is not the maize
    assert m[30] == 1                             # maize in summer


def test_index_carries_a_data_version(tmp_path, monkeypatch):
    # review 2026-10-20: field files are cached for a week; the app appends ?v=<version> so a rebuild never mixes files
    import json
    monkeypatch.setattr(X, "DOCS", str(tmp_path))
    X.write_index()
    v = json.load(open(tmp_path / "index.json"))["version"]
    assert isinstance(v, str) and len(v) >= 8


def test_field_clear_dates_need_half_the_field_clear():
    # owner check 20 Oct: the field panel showed a black image; a date counts for a field only if >= 50 % of it was clear
    clear = np.array([[True, True, False, False],     # date 0: field 0 fully clear, field 1 not
                      [True, False, True, True],      # date 1: field 0 half (by weight 1 of 2), field 1 clear
                      [False, False, True, False]])   # date 2: field 1 half
    w = np.array([1.0, 1.0, 1.0, 1.0])
    inv = np.array([0, 0, 1, 1])
    clr, area = X.clear_dates(clear, w, inv, n_fields=2, min_share=0.5)
    assert clr == [[0, 1], [1, 2]]
    assert np.allclose(area, [0.5, 0.75, 0.25])


def test_field_image_needs_95_percent_clear_by_default():
    clear = np.array([[True] * 19 + [False], [True] * 20])        # 95 % then 100 % of one field
    clr, _ = X.clear_dates(clear, np.ones(20), np.zeros(20, int), n_fields=1)
    assert clr == [[0, 1]]
    clear = np.array([[True] * 18 + [False] * 2])                  # 90 %: not clean enough
    assert X.clear_dates(clear, np.ones(20), np.zeros(20, int), n_fields=1)[0] == [[]]


def test_off_season_spring_cover_crop_greening_in_spring_is_not_the_crop():
    # owner check 20 Oct (north 2025 maize): grass / cover crop greens up again in March-April, is cut in May, maize from June
    lai = [1.0] * 21 + [1.5, 2.5, 3.5, 4.5, 5.0, 4.0, 1.0, 0.2, 0.1, 0.1, 0.3, 0.8, 1.5, 2.5, 3.0, 3.2, 3.0, 2.8] + [2.5] * 9
    m = X.season_mask(lai, "maize")
    assert m[24] == 0                              # April green-up is the cover crop
    assert m[35] == 1                              # maize in July
