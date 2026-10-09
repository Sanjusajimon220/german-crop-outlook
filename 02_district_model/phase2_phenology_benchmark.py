"""Phase 2: gradient-boosting benchmark for the crop calendar (stage dates).

For each crop and stage, gradient boosting predicts the day of year of the stage from the
master-file columns that would be known before the stage: monthly weather only of months that
end before the stage's typical (median) date, the winter summary for winter crops, sowing day,
soil water, latitude and year. Trained on the training rows only, with the setting chosen by
leaving out one year block at a time; scored on the same test rows as the calendar model
(phase2_phenology.py), next to the calendar model's own errors.

Run `python phase2_phenology_benchmark.py [crop ...]` after phase2_phenology.py.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from phase2_phenology import CROPS
from wofost_runner import district_latitudes

P = os.path.join("data", "processed")
SETTINGS = ({"max_depth": 3, "learning_rate": 0.05}, {"max_depth": 6, "learning_rate": 0.05})
ROLES = ("test_years", "test_regions", "test_both")


def known_before(columns, stage_doy):
    """Monthly columns of months ending before the stage's median day (+ non-monthly ones)."""
    keep = []
    for c in columns:
        m = re.fullmatch(r"(tas|pr|rsds|et0|hot)_m(\d\d)", c)
        if m:
            month_end = pd.Timestamp(2001, int(m.group(2)), 1) + pd.offsets.MonthEnd(0)
            if month_end.dayofyear < stage_doy:
                keep.append(c)
        elif c in ("tas_winter", "pr_winter", "frost_days_winter", "paw_1m_mm", "paw_2m_mm",
                   "doy_sowing", "year", "lat"):
            keep.append(c)
    return keep


def model(setting):
    return HistGradientBoostingRegressor(max_iter=300, min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=0, **setting)


def main():
    split = pd.read_csv(os.path.join(P, "split.csv"), dtype={"district": str})
    lat = district_latitudes()
    for crop in (sys.argv[1:] or list(CROPS)):
        stages = CROPS[crop][0]
        m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
        m = m.merge(split, on=["district", "year"])
        m["lat"] = m["district"].map(lat)
        pred_path = os.path.join(P, f"phenology_predictions_{crop}.csv")
        cal = pd.read_csv(pred_path, dtype={"district": str}) if os.path.exists(pred_path) else None
        print(f"\n{crop}: RMSE in days, gradient boosting (calendar model / average of both / simple forecast)")
        print(f"  {'stage':15s}" + "".join(f"{r:>28s}" for r in ROLES))
        for st in stages:
            col = f"doy_{st}"
            data = m[m[col].notna()].copy()
            feats = known_before(list(data.columns), data[col].median())
            train = data[data["role"] == "train"]
            best, best_err = None, np.inf
            for s in SETTINGS:
                err = []
                for f in sorted(train["fold"].unique()):
                    tr, va = train[train["fold"] != f], train[train["fold"] == f]
                    p = model(s).fit(tr[feats], tr[col]).predict(va[feats])
                    err.append((p - va[col].values) ** 2)
                e = np.sqrt(np.concatenate(err).mean())
                if e < best_err:
                    best, best_err = s, e
            fitted = model(best).fit(train[feats], train[col])
            data["boost"] = fitted.predict(data[feats])
            clim = train.groupby("district")[col].median()
            data["clim"] = data["district"].map(clim).fillna(train[col].median())
            if cal is not None:
                data = data.merge(cal[["district", "year", f"pred_{st}"]], on=["district", "year"], how="left")
            line = f"  {st:15s}"
            for role in ROLES:
                sel = data["role"] == role
                if cal is not None:
                    sel &= data[f"pred_{st}"].notna()
                obs = data.loc[sel, col]

                def rmse(x):
                    return np.sqrt(np.mean((x - obs) ** 2))
                if cal is not None:
                    both = (data.loc[sel, "boost"] + data.loc[sel, f"pred_{st}"]) / 2
                    calendar = f"{rmse(data.loc[sel, f'pred_{st}']):4.1f} /{rmse(both):4.1f}"
                else:
                    calendar = "  -  /  - "
                line += f"   {rmse(data.loc[sel, 'boost']):5.1f} ({calendar} /{rmse(data.loc[sel, 'clim']):5.1f})"
            print(line + f"   [{len(feats)} features]", flush=True)


if __name__ == "__main__":
    main()
