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
    rain = lambda d, k, lag: w["pr"][pos[d - pd.Timedelta(days=lag)], col[k]] if (d - pd.Timedelta(days=lag)) in pos and k in col else np.nan
    s["rain0"] = [rain(d, k, 0) for d, k in zip(s["date"], s["district"])]
    s["rain1"] = [rain(d, k, 1) for d, k in zip(s["date"], s["district"])]
    head = {k: v[0] for k, v in district_dates(year).items()}
    s["since_heading"] = [(d - head[k]).days if k in head else np.nan for d, k in zip(s["date"], s["district"])]
    start = pd.Timestamp(f"{year - 1}-09-01")
    s["doy"] = (s["date"] - start).dt.days
    return s


def nearest(s1f, d, tol=2):
    if s1f.empty:
        return None
    k = (s1f["date"] - d).abs()
    i = k.idxmin()
    return s1f.loc[i] if k[i].days <= tol else None


def main(year, n):
    s1 = s1_table(year, n)
    obs = observations(year)[["field_id", "district", "date", "lai"]]
    obs = obs[obs["field_id"].isin(s1["field_id"].unique())]
    dist = obs.groupby("field_id")["district"].first()
    s1 = add_context(s1, year, dist)
    rng = np.random.default_rng(7)
    fids = rng.permutation(obs["field_id"].unique())
    folds = {f: i % 5 for i, f in enumerate(fids)}
    # training pairs: every clear S2 date with an S1 date within +-2 days
    pairs = []
    for fid, g in obs.groupby("field_id"):
        sf = s1[s1["field_id"] == fid]
        for _, o in g.iterrows():
            r = nearest(sf, o["date"])
            if r is not None:
                pairs.append({"field_id": fid, "date": o["date"], "lai": o["lai"], **r[FEATS].to_dict()})
    pairs = pd.DataFrame(pairs)
    pairs["fold"] = pairs["field_id"].map(folds)
    print(f"{year}: {obs['field_id'].nunique()} fields, {len(obs)} clear S2 dates, {len(s1)} S1 field-dates, "
          f"{len(pairs)} S1-S2 pairs (+-2 days)", flush=True)
    rows = []
    for k in range(5):
        tr = pairs[pairs["fold"] != k]
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(tr[FEATS], tr["lai"])
        for fid in [f for f, v in folds.items() if v == k]:
            g = obs[obs["field_id"] == fid].sort_values("date").reset_index(drop=True)
            if len(g) < 6:
                continue
            hide = rng.random(len(g)) < 0.3
            hide[0] = hide[-1] = False
            kept = g[~hide]
            sf = s1[s1["field_id"] == fid]
            for _, o in g[hide].iterrows():
                before, after = kept[kept["date"] < o["date"]], kept[kept["date"] > o["date"]]
                gap = (after["date"].iloc[0] - before["date"].iloc[-1]).days
                interp = np.interp(o["date"].value, kept["date"].astype("int64"), kept["lai"])
                r = nearest(sf, o["date"])
                s1p = float(model.predict(pd.DataFrame([r[FEATS]]))[0]) if r is not None else np.nan
                rows.append(dict(field_id=fid, date=o["date"], lai=o["lai"], interp=interp, s1=s1p, gap=gap,
                                 since_heading=r["since_heading"] if r is not None else np.nan))
    e = pd.DataFrame(rows).dropna(subset=["s1"])
    e["gap_class"] = pd.cut(e["gap"], [0, 15, 30, 400], labels=["<=15 d", "16-30 d", ">30 d"])
    e["phase"] = np.where(e["since_heading"] < 0, "before heading", "after heading")
    rm = lambda x: float(np.sqrt(np.mean(np.square(x))))
    summ = []
    for key, g in [("all", e)] + [(f"gap {c}", g) for c, g in e.groupby("gap_class", observed=True)] + \
                  [(p, g) for p, g in e.groupby("phase")]:
        summ.append(dict(subset=key, n=len(g), rmse_s1=rm(g["s1"] - g["lai"]), rmse_interp=rm(g["interp"] - g["lai"]),
                         r_s1=g["s1"].corr(g["lai"]), r_interp=g["interp"].corr(g["lai"])))
    s = pd.DataFrame(summ).round(3)
    os.makedirs(os.path.join(P, "s1"), exist_ok=True)
    e.to_csv(os.path.join(P, "s1", f"gapfill_check_{year}.csv"), index=False)
    s.to_csv(os.path.join(P, "s1", f"gapfill_summary_{year}.csv"), index=False)
    print(s.to_string(index=False))
    long = s[s["subset"].isin(["gap 16-30 d", "gap >30 d"])]
    ok = len(long) and (long["rmse_s1"] < long["rmse_interp"]).all()
    print("RESULT:", "ADOPT (S1 beats interpolation on gaps > 15 days)" if ok else "NOT adopted (rule not met)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
