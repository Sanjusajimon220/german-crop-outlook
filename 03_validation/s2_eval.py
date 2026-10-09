"""v11.1 Sentinel-2 test, exactly as pre-registered (docs/project_log.md, 2026-10-09 16:25). Run ONCE when the
Sentinel-2 years 2019-2023 are complete.

Target v10 district residual (official - median, ranges_v10_final); ridge (alpha 10) on standardised S2 features
(peak, last3, integral + state anomalies); leave-one-year-out; held-out states 01/08/12 never in fitting;
correction = 0.5 * prediction. Output data/processed/s2_eval.csv + adoption flags.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = os.path.join("data", "processed")
F = ["peak", "last3", "integral", "peak_anom", "last3_anom", "integral_anom"]
HELD = {"01", "08", "12"}


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    s2 = pd.read_csv(os.path.join(P, "s2_features.csv"), dtype={"district": str, "state": str})
    area = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    area = area[area["area_ha"] > 0].set_index(["crop", "district", "year"])["area_ha"]
    out = []
    for (crop, lead), g in s2.groupby(["crop", "lead"]):
        f = os.path.join(P, "ranges_v10_final", f"ranges_{crop}_lead{lead:02d}.csv")
        if not os.path.exists(f):
            continue
        v = pd.read_csv(f, dtype={"region": str})
        v = v[v["level"] == "district"].rename(columns={"region": "district"})[["district", "year", "obs", "median"]]
        d = g.merge(v, on=["district", "year"]).dropna(subset=["obs", "median"] + F)
        d["res"] = d["obs"] - d["median"]
        d["area"] = [area.get((crop, a, y), np.nan) for a, y in zip(d["district"], d["year"])]
        years = sorted(d["year"].unique())
        if len(years) < 4:
            continue
        d["corr"] = np.nan
        for y in years:
            tr = d[(d["year"] != y) & ~d["state"].isin(HELD)]
            te = d["year"] == y
            sc = StandardScaler().fit(tr[F])
            m = Ridge(alpha=10).fit(sc.transform(tr[F]), tr["res"])
            d.loc[te, "corr"] = 0.5 * m.predict(sc.transform(d.loc[te, F]))
        d["new"] = d["median"] + d["corr"]
        known = ~d["state"].isin(HELD)
        r = dict(crop=crop, lead=lead, n=len(d), years=len(years))
        per = []
        for y in years:
            k = known & (d["year"] == y)
            per.append(rmse(d.loc[k, "new"] - d.loc[k, "obs"]) < rmse(d.loc[k, "median"] - d.loc[k, "obs"]))
        r["dist_old"] = rmse(d.loc[known, "median"] - d.loc[known, "obs"])
        r["dist_new"] = rmse(d.loc[known, "new"] - d.loc[known, "obs"])
        r["dist_gain_pct"] = 100 * (1 - r["dist_new"] / r["dist_old"])
        r["years_better"] = int(sum(per))
        h = ~known
        r["held_old"] = rmse(d.loc[h, "median"] - d.loc[h, "obs"]) if h.any() else np.nan
        r["held_new"] = rmse(d.loc[h, "new"] - d.loc[h, "obs"]) if h.any() else np.nan
        a = d.dropna(subset=["area"])
        for lvl, key in (("state", a["state"]), ("nat", pd.Series("DE", index=a.index))):
            agg = a.assign(k=key).groupby(["k", "year"]).apply(
                lambda x: pd.Series({c: np.average(x[c], weights=x["area"]) for c in ("obs", "median", "new")}))
            r[f"{lvl}_old"] = rmse(agg["median"] - agg["obs"])
            r[f"{lvl}_new"] = rmse(agg["new"] - agg["obs"])
        need = len(years) - 1
        r["adopt"] = bool(r["dist_gain_pct"] >= 3 and r["years_better"] >= need
                          and (np.isnan(r["held_old"]) or r["held_new"] <= r["held_old"])
                          and r["state_new"] <= r["state_old"] and r["nat_new"] <= r["nat_old"])
        out.append(r)
    o = pd.DataFrame(out)
    o.to_csv(os.path.join(P, "s2_eval.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(o.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
