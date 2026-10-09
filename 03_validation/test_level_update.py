"""Pre-registered annual level update test (project log 2026-10-09), training years only.
Forward predictions of the current v8 model: period A = fit <= 2002, years 2003-2008; period B = fit <= 2008,
years 2009-2017. For forecast year t: national correction = mean of the yearly mean errors of the last
3 (N3) or 5 (N5) earlier years; district offset = shrunk mean (k = 3) of district errors (year part
removed) of years <= t-2. Period B may use period-A errors as history."""
import os
import numpy as np
import pandas as pd
P = os.path.join("data", "processed")
K = 3
BASE = {"potato": "_potato_canopy"}
rows = []
for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
    b = BASE.get(crop, "")
    A = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_v8{b}_cut2002.csv"), dtype={"district": str})
    A = A[A["year"] <= 2008].dropna(subset=["yield_t_ha", "pred"]).assign(period="A")
    B = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_v8{b}_cut2008.csv"), dtype={"district": str})
    B = B.dropna(subset=["yield_t_ha", "pred"]).assign(period="B")
    allf = pd.concat([A, B], ignore_index=True)
    allf["e"] = allf["pred"] - allf["yield_t_ha"]
    ye = allf.groupby("year")["e"].mean()
    allf["e_loc"] = allf["e"] - allf["year"].map(ye)
    res = {v: [] for v in ("none", "D", "N3", "N5", "N3+D", "N5+D")}
    for per, g in allf.groupby("period"):
        hist_years = sorted(allf["year"].unique()) if per == "B" else sorted(A["year"].unique())
        for t, gt in g.groupby("year"):
            prev = [y for y in hist_years if y < t]
            n3 = ye.loc[prev[-3:]].mean() if prev else 0.0
            n5 = ye.loc[prev[-5:]].mean() if prev else 0.0
            h = allf[(allf["year"] <= t - 2) & ((allf["period"] == "A") | (per == "B"))]
            h = h[h["year"].isin(prev)]
            off = (h.groupby("district")["e_loc"].sum() / (h.groupby("district")["e_loc"].count() + K)) if len(h) else pd.Series(dtype=float)
            d = gt["district"].map(off).fillna(0).values
            e = gt["e"].values
            for v, corr in (("none", 0), ("D", d), ("N3", n3), ("N5", n5), ("N3+D", n3 + d), ("N5+D", n5 + d)):
                res[v] += [(per, x) for x in (e - corr)]
    for v, lst in res.items():
        r = pd.DataFrame(lst, columns=["period", "e"])
        rows.append(dict(crop=crop, variant=v, **{f"rmse_{p}": np.sqrt((g["e"] ** 2).mean()) for p, g in r.groupby("period")},
                         **{f"bias_{p}": g["e"].mean() for p, g in r.groupby("period")}))
s = pd.DataFrame(rows)
out = []
for crop, g in s.groupby("crop", sort=False):
    base = g[g["variant"] == "none"].iloc[0]
    g = g.assign(chg_A=100 * (g["rmse_A"] / base["rmse_A"] - 1), chg_B=100 * (g["rmse_B"] / base["rmse_B"] - 1))
    ok = g[(g["chg_A"] < 0) & (g["chg_B"] < 0)].assign(mean=lambda x: (x["rmse_A"] + x["rmse_B"]) / 2).sort_values("mean")
    order = {"D": 0, "N3": 1, "N5": 2, "N3+D": 3, "N5+D": 3}
    choice = "none"
    if len(ok):
        near = ok[ok["mean"] <= ok["mean"].iloc[0] * 1.01]
        choice = min(near["variant"], key=lambda v: order[v])
    out.append(g.assign(chosen=g["variant"] == choice))
s = pd.concat(out).round(3)
s.to_csv(os.path.join(P, "level_update_test.csv"), index=False)
with pd.option_context("display.width", 200):
    print(s.to_string(index=False))
