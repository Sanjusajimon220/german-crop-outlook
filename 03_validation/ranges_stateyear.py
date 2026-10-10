"""v11.1 item 3: state-year component in the 80 % ranges (pre-registered docs/project_log.md 2026-10-10 13:45).
TRAINING years 2011-2017 only (v9 fwd2008 in-season forecasts), framework of ranges_adaptive.py.

current : sd^2 = weather^2 + year^2 + local^2                  (state: local^2 x sum w^2)
stateyr : local error split into state-year mean (sy) + rest; sd^2 = weather^2 + year^2 + sy^2 + rest^2
          state: (sum w weather)^2 + year^2 + sy^2 + rest^2 x sum w^2; national: sy^2 x sum W_state^2.
Mode per crop as decided for v10: adaptive (barley, grain maize) / static (others).
Writes data/processed/ranges_stateyear_train.csv and prints coverage / interval score per crop and level.
"""
import os

import numpy as np
import pandas as pd

from phase5_benchmarks import census_area
from ranges_adaptive import F, P, Z, history, iscore

ADAPTIVE = {"winter_barley", "grain_maize"}
CROPS = ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato")


def sds(h, ym, t, adaptive):
    yh = ym[ym.index < t]
    old = h[h["year"] <= t - 2]
    ys, ls, ss, rs = yh.std(ddof=1), old["loc"].std(), old.groupby(["st", "year"])["sy"].first().std(), old["rest"].std()
    if adaptive:
        ry = ym[(ym.index >= t - 5) & (ym.index <= t - 1)]
        rec = h[(h["year"] >= t - 6) & (h["year"] <= t - 2)]
        rms = lambda x: np.sqrt((x ** 2).mean())
        ys, ls = max(ys, rms(ry)), max(ls, rms(rec["loc"]))
        ss, rs = max(ss, rms(rec.groupby(["st", "year"])["sy"].first())), max(rs, rms(rec["rest"]))
    return ys, ls, ss, rs


def main():
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    rows = []
    for crop in CROPS:
        yc = y[y["crop"] == crop].copy()
        yc["area"] = census_area(yc)
        area = yc.set_index(["district", "year"])["area"]
        ym, h = history(crop)
        h["st"] = h["district"].str[:2]
        h["sy"] = h.groupby(["st", "year"])["loc"].transform("mean")
        h["rest"] = h["loc"] - h["sy"]
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
                g["a"] = [area.get((d, t), np.nan) for d in g["district"]]
                for mode in ("current", "stateyr"):
                    loc2 = ls ** 2 if mode == "current" else ss ** 2 + rs ** 2
                    sd = np.sqrt(g["wsd"] ** 2 + ys ** 2 + loc2)
                    lo, hi = g[0.5] - Z * sd, g[0.5] + Z * sd
                    rows.append(dict(crop=crop, lead=lead, year=t, level="district", mode=mode,
                                     inside=float(((g["obs"] >= lo) & (g["obs"] <= hi)).mean()),
                                     iscore=float(iscore(lo, hi, g["obs"]).mean()), n=len(g)))
                    ga = g.dropna(subset=["a"])
                    ga = ga[ga["a"] > 0]
                    st = ga["district"].str[:2].values
                    for level, keys in (("national", np.array(["DE"] * len(ga))), ("state", st)):
                        for k in np.unique(keys):
                            s = ga[keys == k]
                            if len(s) < 3:
                                continue
                            wt = s["a"] / s["a"].sum()
                            med, obs = float((wt * s[0.5]).sum()), float((wt * s["obs"]).sum())
                            w2 = float((wt ** 2).sum())
                            if mode == "current":
                                var = ls ** 2 * w2
                            elif level == "state":
                                var = ss ** 2 + rs ** 2 * w2
                            else:
                                W = wt.groupby(s["district"].str[:2].values).sum()
                                var = ss ** 2 * float((W ** 2).sum()) + rs ** 2 * w2
                            sdn = np.sqrt(float((wt * s["wsd"]).sum()) ** 2 + ys ** 2 + var)
                            lo_, hi_ = med - Z * sdn, med + Z * sdn
                            rows.append(dict(crop=crop, lead=lead, year=t, level=level, mode=mode,
                                             inside=float(lo_ <= obs <= hi_), iscore=float(iscore(lo_, hi_, obs)), n=1))
        print(crop, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(P, "ranges_stateyear_train.csv"), index=False)
    s = r.groupby(["crop", "level", "mode"]).apply(lambda g: pd.Series({
        "coverage80": np.average(g["inside"], weights=g["n"]), "iscore": np.average(g["iscore"], weights=g["n"])}),
        include_groups=False).unstack("mode").round(3)
    with pd.option_context("display.width", 200):
        print(s.to_string())


if __name__ == "__main__":
    main()
