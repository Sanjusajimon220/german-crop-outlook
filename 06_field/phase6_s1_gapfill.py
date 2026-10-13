"""Sentinel-1 gap filling for green LAI (pre-registered docs/project_log.md 2026-10-13). Sentinel-2 stays primary;
Sentinel-1 only fills cloudy gaps.

Inputs  data/processed/s1/s1_115_<year>_<n>fields.csv (field x date: VV, VH linear sigma0)
        data/processed/s2/retrieval_115_<year>.csv (field x clear date: green LAI from PROSAIL network)
Model   gradient boosting: S1 features -> S2 green LAI, trained on pairs within +-2 days.
        features: VV_db, VH_db, VH-VV, their change since the field's previous S1 date, day of season,
        district rain on the day and the day before, days since district heading (calendar).
Check   5 folds by FIELD. In the test fields, 30 % of the clear S2 dates are hidden; each hidden value is predicted
        (a) by S1 (nearest S1 date within +-2 days), (b) by linear interpolation between the kept S2 dates around it.
        RMSE / r by gap length (days between the kept neighbours) and by phase. Adopt if S1 beats interpolation on
        gaps > 15 days.
Run `python phase6_s1_gapfill.py 2024 300`.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from phase4_field_assim import observations
from phase4_field_features import district_dates

P = os.path.join("data", "processed")
FEATS = ["VV_db", "VH_db", "ratio_db", "dVV", "dVH", "dratio", "doy", "rain0", "rain1", "since_heading"]


def s1_table(year, n):
    s = pd.read_csv(os.path.join(P, "s1", f"s1_115_{year}_{n}fields.csv"))
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None).dt.normalize()
    s = s.dropna(subset=["VV", "VH"])
    s = s[(s["VV"] > 0) & (s["VH"] > 0)].sort_values(["field_id", "date"])
    s["VV_db"], s["VH_db"] = 10 * np.log10(s["VV"]), 10 * np.log10(s["VH"])
    s["ratio_db"] = s["VH_db"] - s["VV_db"]
    for a, b in (("dVV", "VV_db"), ("dVH", "VH_db"), ("dratio", "ratio_db")):
        s[a] = s.groupby("field_id")[b].diff()
    return s.groupby(["field_id", "date"], as_index=False).mean(numeric_only=True)


def add_context(s, year, dist):
    w = np.load(os.path.join(P, "weather_daily_v8.npz"))
    dates = pd.DatetimeIndex(w["dates"])
    col = {c: i for i, c in enumerate(w["codes"])}
    pos = {d: i for i, d in enumerate(dates)}
    s["district"] = s["field_id"].map(dist)
    s = s.dropna(subset=["district"])
    pr = w["pr"]                     # read once (an npz re-reads the whole array on every access)
    rain = lambda d, k, lag: pr[pos[d - pd.Timedelta(days=lag)], col[k]] if (d - pd.Timedelta(days=lag)) in pos and k in col else np.nan
    s["rain0"] = [rain(d, k, 0) for d, k in zip(s["date"], s["district"])]
    s["rain1"] = [rain(d, k, 1) for d, k in zip(s["date"], s["district"])]
    head = {k: v[0] for k, v in district_dates(year).items()}
    s["since_heading"] = [(d - head[k]).days if k in head else np.nan for d, k in zip(s["date"], s["district"])]
    start = pd.Timestamp(f"{year - 1}-09-01")
    s["doy"] = (s["date"] - start).dt.days
    return s


def attach_s1(df, s1, tol=2):
    """Nearest S1 date per row within +-tol days (vectorised merge_asof per field)."""
    out = pd.merge_asof(df.sort_values("date"), s1[["field_id", "date"] + FEATS].rename(columns={"date": "s1_date"}).sort_values("s1_date"),
                        left_on="date", right_on="s1_date", by="field_id", direction="nearest",
                        tolerance=pd.Timedelta(days=tol))
    return out


def main(year, n):
    s1 = s1_table(year, n)
    obs = observations(year)[["field_id", "district", "date", "lai"]]
    obs = obs[obs["field_id"].isin(s1["field_id"].unique())].copy()
    dist = obs.groupby("field_id")["district"].first()
    s1 = add_context(s1, year, dist)
    rng = np.random.default_rng(7)
    fids = rng.permutation(obs["field_id"].unique())
    folds = {f: i % 5 for i, f in enumerate(fids)}
    obs["fold"] = obs["field_id"].map(folds)
    pairs = attach_s1(obs, s1).dropna(subset=["s1_date"])
    print(f"{year}: {obs['field_id'].nunique()} fields, {len(obs)} clear S2 dates, {len(s1)} S1 field-dates, "
          f"{len(pairs)} S1-S2 pairs (+-2 days)", flush=True)
    # hidden dates: 30 % per field, never the first or last date of a field
    obs = obs.sort_values(["field_id", "date"]).reset_index(drop=True)
    obs["hide"] = rng.random(len(obs)) < 0.3
    first = obs.groupby("field_id")["date"].transform("min") == obs["date"]
    last = obs.groupby("field_id")["date"].transform("max") == obs["date"]
    obs.loc[first | last, "hide"] = False
    obs = obs[obs.groupby("field_id")["date"].transform("size") >= 6]
    rows = []
    for fid, g in obs.groupby("field_id"):
        kept, hid = g[~g["hide"]], g[g["hide"]]
        if hid.empty:
            continue
        kd = kept["date"].values.astype("int64")
        interp = np.interp(hid["date"].values.astype("int64"), kd, kept["lai"].values)
        idx = np.searchsorted(kd, hid["date"].values.astype("int64"))
        gap = (kd[idx] - kd[idx - 1]) / 8.64e13
        rows.append(hid.assign(interp=interp, gap=gap))
    e = pd.concat(rows)
    e = attach_s1(e, s1).dropna(subset=["s1_date"])
    e["s1"] = np.nan
    for k in range(5):
        tr = pairs[pairs["fold"] != k]
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(tr[FEATS], tr["lai"])
        m = e["fold"] == k
        if m.any():
            e.loc[m, "s1"] = model.predict(e.loc[m, FEATS])
    e["gap_class"] = pd.cut(e["gap"], [0, 15, 30, 400], labels=["<=15 d", "16-30 d", ">30 d"])
    e["phase"] = np.where(e["since_heading"] < 0, "before heading", "after heading")
    rm = lambda x: float(np.sqrt(np.mean(np.square(x))))
    summ = []
    for key, g in [("all", e)] + [(f"gap {c}", g) for c, g in e.groupby("gap_class", observed=True)] +                   [(p, g) for p, g in e.groupby("phase")]:
        summ.append(dict(subset=key, n=len(g), rmse_s1=rm(g["s1"] - g["lai"]), rmse_interp=rm(g["interp"] - g["lai"]),
                         r_s1=g["s1"].corr(g["lai"]), r_interp=g["interp"].corr(g["lai"])))
    s = pd.DataFrame(summ).round(3)
    e.to_csv(os.path.join(P, "s1", f"gapfill_check_{year}.csv"), index=False)
    s.to_csv(os.path.join(P, "s1", f"gapfill_summary_{year}.csv"), index=False)
    print(s.to_string(index=False))
    long = s[s["subset"].isin(["gap 16-30 d", "gap >30 d"])]
    ok = len(long) and (long["rmse_s1"] < long["rmse_interp"]).all()
    print("RESULT:", "ADOPT (S1 beats interpolation on gaps > 15 days)" if ok else "NOT adopted (rule not met)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
