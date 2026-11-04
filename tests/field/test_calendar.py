import numpy as np
import pandas as pd

from phase25_calendar import calendar_crop, nearest_assign, stage_dates, weekly_dev


def test_stage_dates_are_the_first_day_the_curve_reaches_each_threshold():
    dev = np.array([0.0, 2.0, 5.0, 9.0, 14.0, 20.0])            # day k after sowing
    out = stage_dates(dev, np.array([4.0, 9.0, 30.0]), pd.Timestamp("2024-10-01"), ["emergence", "heading", "ripe"])
    assert out == {"emergence": "2024-10-03", "heading": "2024-10-04", "ripe": None}   # never reached -> None


def test_weekly_dev_samples_mondays_and_leaves_gaps_outside_the_run():
    dev = np.arange(10, dtype=float) * 1.25                       # 10 days from sowing
    weeks = np.array(["2024-09-30", "2024-10-07", "2024-10-14"], dtype="datetime64[D]")
    out = weekly_dev(dev, pd.Timestamp("2024-10-02"), weeks)
    assert out == [None, 6.2, None]                              # before sowing; day 5; beyond the horizon


def test_calendar_crop_maps_register_names_to_fitted_calendars():
    assert calendar_crop("Winterweichweizen") == "wheat"
    assert calendar_crop("Wintergerste") == "barley"
    for other in ("Silomais", "Mais (ohne Silomais)", "Kartoffeln", "Mais-Mischkulturen", "Sommergerste", "Winter-Dinkel",
                  "Zuckerrüben", "Winterraps", None):
        assert calendar_crop(other) is None                                 # no calendar shown (maize / potato removed)


def test_nearest_assign_takes_cell_and_district_of_the_nearest_reference_field_within_reach():
    ref_xy = np.array([[0.0, 0.0], [5000.0, 0.0]])
    cells, dists = np.array([11, 22]), np.array(["05358", "05370"])
    c, d = nearest_assign(np.array([[300.0, 400.0], [4900.0, 0.0], [2500.0, 3000.0]]), ref_xy, cells, dists, max_m=1500)
    assert c.tolist() == [11, 22, -1] and d.tolist() == ["05358", "05370", None]   # third: nothing within 1.5 km
