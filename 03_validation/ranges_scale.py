"""v12 B: scale factor on the range sd per crop x level (pre-registered docs/project_log.md 2026-10-10 18:28).
TRAINING years 2011-2017 (v9 fwd2008 in-season forecasts), v11.1 range method (state level with state-year part).
k = 80th percentile of |error / sd| / 1.2816 from one half (2011-2014 or 2015-2017), applied to the other half.
Adopt per crop x level if, in BOTH directions, coverage moves closer to 80 % and the interval score is lower.
Writes data/processed/ranges_scale_train.csv and ranges_scale_k.csv (k from all 2011-2017 for adopted cells).
"""
import os

import numpy as np
import pandas as pd

from phase5_benchmarks import census_area
from ranges_adaptive import F, P, Z, history, iscore
from ranges_stateyear import ADAPTIVE, CROPS, sds


def rows_for(crop):
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    yc = y[y["crop"] == crop].copy()
    yc["area"] = census_area(yc)
    area = yc.set_index(["district", "year"])["area"]
    ym, h = history(crop)
    h["st"] = h["district"].str[:2]
    h["sy"] = h.groupby(["st", "year"])["loc"].transform("mean")
    h["rest"] = h["loc"] - h["sy"]
    out = []
    for lead in (12, 8, 6, 4, 2, 0):
        f = pd.read_csv(os.path.join(F, f"forecast_{crop}_v9_fwd2008_lead{lead:02d}.csv"), dtype={"district": str})
        q = f.groupby(["district", "year"])["blend"].quantile([0.1, 0.5, 0.9]).unstack().reset_index()
        q = q.merge(f.groupby(["district", "year"])["obs"].first().reset_index(), on=["district", "year"])
        q["wsd"] = (q[0.9] - q[0.1]) / (2 * Z)
        for t in range(2011, 2018):
            g = q[q["year"] == t].dropna(subset=["obs"]).copy()
            if g.empty:
                continue
            ys, ls, ss, rs = sds(h, ym, t, crop in ADAPTIVE)
            sd = np.sqrt(g["wsd"] ** 2 + ys ** 2 + ls ** 2)
            out += [dict(lead=lead, year=t, level="district", med=m, sd=s, obs=o) for m, s, o in zip(g[0.5], sd, g["obs"])]
            g["a"] = [area.get((d, t), np.nan) for d in g["district"]]
            ga = g.dropna(subset=["a"])
            ga = ga[ga["a"] > 0]
            st = ga["district"].str[:2].values
            for level, keys in (("national", np.array(["DE"] * len(ga))), ("state", st)):
                for k in np.unique(keys):
                    s = ga[keys == k]
                    if len(s) < 3:
                        continue
                    wt = s["a"] / s["a"].sum()
                    w2 = float((wt ** 2).sum())
                    var = ss ** 2 + rs ** 2 * w2 if level == "state" else ls ** 2 * w2
                    sdn = np.sqrt(float((wt * s["wsd"]).sum()) ** 2 + ys ** 2 + var)
                    out.append(dict(lead=lead, year=t, level=level, med=float((wt * s[0.5]).sum()), sd=sdn,
                                    obs=float((wt * s["obs"]).sum())))
    return pd.DataFrame(out).assign(crop=crop)


def score(g, k):
    lo, hi = g["med"] - Z * k * g["sd"], g["med"] + Z * k * g["sd"]
    return float(((g["obs"] >= lo) & (g["obs"] <= hi)).mean()), float(iscore(lo, hi, g["obs"]).mean())


def main():
    d = pd.concat([rows_for(c) for c in CROPS], ignore_index=True)
    d["z"] = (d["obs"] - d["med"]).abs() / d["sd"]
    res, ks = [], []
    for (crop, level), g in d.groupby(["crop", "level"]):
        a, b = g[g["year"] <= 2014], g[g["year"] >= 2015]
        ok = True
        for fit, test, name in ((a, b, "2011-14 -> 2015-17"), (b, a, "2015-17 -> 2011-14")):
            k = float(np.quantile(fit["z"], 0.8) / Z)
            c0, i0 = score(test, 1.0)
            c1, i1 = score(test, k)
            better = abs(c1 - 0.8) < abs(c0 - 0.8) and i1 < i0
            ok &= better
            res.append(dict(crop=crop, level=level, direction=name, k=round(k, 3), cov_before=round(c0, 3), cov_after=round(c1, 3),
                            iscore_before=round(i0, 3), iscore_after=round(i1, 3), better=better))
        ks.append(dict(crop=crop, level=level, k_all=round(float(np.quantile(g["z"], 0.8) / Z), 3), adopt=bool(ok)))
    r, k = pd.DataFrame(res), pd.DataFrame(ks)
    r.to_csv(os.path.join(P, "ranges_scale_train.csv"), index=False)
    k.to_csv(os.path.join(P, "ranges_scale_k.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(r.to_string(index=False)); print(k.to_string(index=False))


if __name__ == "__main__":
    main()
