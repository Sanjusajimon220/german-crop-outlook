import numpy as np
import pandas as pd

from phase25_calendar import stage_dates, weekly_dev


def test_stage_dates_are_the_first_day_the_curve_reaches_each_threshold():
    dev = np.array([0.0, 2.0, 5.0, 9.0, 14.0, 20.0])            # day k after sowing
    out = stage_dates(dev, np.array([4.0, 9.0, 30.0]), pd.Timestamp("2024-10-01"), ["emergence", "heading", "ripe"])
    assert out == {"emergence": "2024-10-03", "heading": "2024-10-04", "ripe": None}   # never reached -> None


def test_weekly_dev_samples_mondays_and_leaves_gaps_outside_the_run():
    dev = np.arange(10, dtype=float) * 1.25                       # 10 days from sowing
    weeks = np.array(["2024-09-30", "2024-10-07", "2024-10-14"], dtype="datetime64[D]")
    out = weekly_dev(dev, pd.Timestamp("2024-10-02"), weeks)
    assert out == [None, 6.2, None]                              # before sowing; day 5; beyond the horizon
