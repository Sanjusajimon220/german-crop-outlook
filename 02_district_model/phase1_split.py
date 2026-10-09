"""Phase 1: the fixed split of the data into training and final test.

Decided on 3 October 2026; the test part is never to be changed afterwards:
  test years    2018-2025 (any federal state)        -> proves forecasting of unseen years
  test regions  Schleswig-Holstein (01), Baden-Wuerttemberg (08), Brandenburg (12), any year
                                                       -> proves transfer to unseen regions
  training      1979-2017 in the other 13 federal states (extended from 1999 the same day,
                when the 1979-2021 yields were added; the test part stayed the same)
Inside the training data, tuning uses six blocks of years (leave one block out at a time),
so the final test data are never looked at before the very end.

Output data/processed/split.csv: district, year, role, fold
  role  train | test_years | test_regions | test_both (test state and test year)
  fold  1-6 for training rows (year blocks 1979-1985, 1986-1991, 1992-1997, 1998-2003,
        2004-2010, 2011-2017), else 0

Run `python phase1_split.py`.
"""
import os

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
TEST_STATES = {"01": "Schleswig-Holstein", "08": "Baden-Wuerttemberg", "12": "Brandenburg"}
TEST_FROM = 2018
FOLDS = [(1979, 1985), (1986, 1991), (1992, 1997), (1998, 2003), (2004, 2010), (2011, 2017)]


def role(district, year):
    region = district[:2] in TEST_STATES
    later = year >= TEST_FROM
    return {(False, False): "train", (False, True): "test_years",
            (True, False): "test_regions", (True, True): "test_both"}[(region, later)]


def fold(year):
    return next((k + 1 for k, (a, b) in enumerate(FOLDS) if a <= year <= b), 0)


def main():
    codes = np.load(os.path.join(P, "weather_daily.npz"))["codes"]
    rows = [(c, y, role(c, y), fold(y) if role(c, y) == "train" else 0)
            for c in codes for y in range(1979, 2026)]
    split = pd.DataFrame(rows, columns=["district", "year", "role", "fold"])
    split.to_csv(os.path.join(P, "split.csv"), index=False)
    print(f"split saved: {len(split)} district-years, "
          f"{split[split.district.str[:2].isin(TEST_STATES)].district.nunique()} districts in test states")
    for crop in ("winter_wheat", "winter_barley", "silage_maize", "grain_maize", "potato"):
        m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str},
                        usecols=["district", "year"])
        r = m.merge(split, on=["district", "year"])["role"].value_counts()
        total = r.sum()
        print(f"  {crop:14s} " + "  ".join(f"{k} {r.get(k, 0)} ({r.get(k, 0) / total:.0%})"
                                           for k in ("train", "test_years", "test_regions", "test_both")))


if __name__ == "__main__":
    main()
