"""S1 gap filling, leave-year-out test (pre-registered, docs/project_log.md 2026-10-13): models trained only on 2024
pairs (same 8 x 8 km cell), tested on the 2025 fields of that cell with the same hiding protocol."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import phase6_s1_gapfill as g

BASE = ["doy", "since_heading", "rain0", "rain1"]


def prepare(year):
    s1 = g.s1_table(year, 300)
    obs = g.observations(year)[["field_id", "district", "date", "lai"]]
    obs = obs[obs["field_id"].isin(s1["field_id"].unique())].copy()
    s1 = g.add_context(s1, year, obs.groupby("field_id")["district"].first())
    return s1, obs


s1a, obsa = prepare(2024)
train = g.attach_s1(obsa, s1a).dropna(subset=["s1_date"])
full = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(train[g.FEATS], train["lai"])
base = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(train[BASE], train["lai"])
s1b, obs = prepare(2025)
rng = np.random.default_rng(7)
obs = obs.sort_values(["field_id", "date"]).reset_index(drop=True)
obs["hide"] = rng.random(len(obs)) < 0.3
first = obs.groupby("field_id")["date"].transform("min") == obs["date"]
last = obs.groupby("field_id")["date"].transform("max") == obs["date"]
obs.loc[first | last, "hide"] = False
obs = obs[obs.groupby("field_id")["date"].transform("size") >= 6]
rows = []
for fid, gg in obs.groupby("field_id"):
    kept, hid = gg[~gg["hide"]], gg[gg["hide"]]
    if hid.empty:
        continue
    kd = kept["date"].values.astype("int64")
    idx = np.searchsorted(kd, hid["date"].values.astype("int64"))
    rows.append(hid.assign(interp=np.interp(hid["date"].values.astype("int64"), kd, kept["lai"].values),
                           gap=(kd[idx] - kd[idx - 1]) / 8.64e13))
e = g.attach_s1(pd.concat(rows), s1b).dropna(subset=["s1_date"])
e["s1"], e["base"] = full.predict(e[g.FEATS]), base.predict(e[BASE])
e["gap_class"] = pd.cut(e["gap"], [0, 15, 30, 400], labels=["<=15 d", "16-30 d", ">30 d"])
rm = lambda x: float(np.sqrt(np.mean(np.square(x))))
print(f"train 2024: {len(train)} pairs; test 2025: {obs['field_id'].nunique()} fields, {len(e)} hidden dates with an S1 date")
out = []
for key, d in [("all", e)] + [(f"gap {c}", d) for c, d in e.groupby("gap_class", observed=True)]:
    out.append(dict(subset=key, n=len(d), rmse_radar=rm(d["s1"] - d["lai"]), rmse_date_only=rm(d["base"] - d["lai"]),
                    rmse_interp=rm(d["interp"] - d["lai"]), r_radar=d["s1"].corr(d["lai"])))
s = pd.DataFrame(out).round(3)
print(s.to_string(index=False))
sub = e.groupby("date").filter(lambda x: len(x) >= 8)
print("same-date ranking r, median: radar", round(sub.groupby("date").apply(lambda x: x["s1"].corr(x["lai"])).median(), 2),
      "date-only", round(sub.groupby("date").apply(lambda x: x["base"].corr(x["lai"])).median(), 2))
long = s[s["subset"].isin(["gap 16-30 d", "gap >30 d"])]
ok = (long["rmse_radar"] < long["rmse_interp"]).all() and s.loc[s["subset"] == "all", "rmse_radar"].iloc[0] < s.loc[s["subset"] == "all", "rmse_date_only"].iloc[0]
print("RESULT:", "ADOPT for all fields" if ok else "NOT met: gap filling stays a within-year tool")
e.to_csv("data/processed/s1/gapfill_loyo_2025.csv", index=False)
s.to_csv("data/processed/s1/gapfill_loyo_summary_2025.csv", index=False)
