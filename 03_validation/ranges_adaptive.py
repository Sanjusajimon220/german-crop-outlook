"""v11 C: adaptive ranges vs static ranges (pre-registered 2026-10-09), TRAINING years only.

Forecasts: in-season forecasts of 2009-2017 from the v9 forward models (forecast_<crop>_v9_fwd2008_leadNN.csv).
Error history (end-of-season, out-of-sample): v8 rolling forward cuts for 1991-2008, v9 forward lead-0
medians for 2009 onwards. For year t: year-wide errors (mean over districts) known up to t-1, local errors
(district minus year mean) up to t-2.
Distribution: median + normal with sd^2 = weather sd^2 (scenario 10-90 % / 2.563) + year sd^2 + local sd^2.
Static: sds from all history before t. Adaptive: max(static, RMS of the last 5 available years).
National / state: area-weighted (latest census area); weather sd added linearly (shared scenarios), year sd
shared, local variance x sum(weights^2). Scores 2011-2017: 80 % coverage, interval score (alpha 0.2).
Writes data/processed/ranges_adaptive_train.csv.
"""
import os

import numpy as np
import pandas as pd

from phase5_benchmarks import census_area

P = os.path.join("data", "processed")
F = os.path.join(P, "forecast")
Z = 1.2816
import sys
CROPS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato")


def history(crop):
    b = "_potato_canopy" if crop == "potato" else ""
    parts = []
    for cut, lo, hi in ((1990, 1991, 1996), (1996, 1997, 2002), (2002, 2003, 2008)):
        f = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_v8{b}_cut{cut}.csv"), dtype={"district": str})
        parts.append(f[f["year"].between(lo, hi)])
    h = pd.concat(parts).dropna(subset=["yield_t_ha", "pred"])
    h = pd.DataFrame({"district": h["district"], "year": h["year"], "e": h["pred"] - h["yield_t_ha"]})
    f0 = pd.read_csv(os.path.join(F, f"forecast_{crop}_v9_fwd2008_lead00.csv"), dtype={"district": str})
    q = f0.groupby(["district", "year"]).agg(med=("blend", "median"), obs=("obs", "first")).reset_index().dropna()
    h = pd.concat([h, pd.DataFrame({"district": q["district"], "year": q["year"], "e": q["med"] - q["obs"]})], ignore_index=True)
    ym = h.groupby("year")["e"].mean()
    h["loc"] = h["e"] - h["year"].map(ym)
    return ym, h


def iscore(lo, hi, y, a=0.2):
    return (hi - lo) + 2 / a * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def main():
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    rows = []
    for crop in CROPS:
        yc = y[y["crop"] == crop].copy()
        yc["area"] = census_area(yc)
        area = yc.set_index(["district", "year"])["area"]
        ym, h = history(crop)
        for lead in (12, 8, 6, 4, 2, 0):
            f = pd.read_csv(os.path.join(F, f"forecast_{crop}_v9_fwd2008_lead{lead:02d}.csv"), dtype={"district": str})
            q = f.groupby(["district", "year"])["blend"].quantile([0.1, 0.5, 0.9]).unstack().reset_index()
            q = q.merge(f.groupby(["district", "year"])["obs"].first().reset_index(), on=["district", "year"])
            q["wsd"] = (q[0.9] - q[0.1]) / (2 * Z)
            for t in range(2011, 2018):
                g = q[q["year"] == t].dropna(subset=["obs"]).copy()
                if g.empty:
                    continue
                yh = ym[ym.index < t]
                lh = h[h["year"] <= t - 2]["loc"]
                ys, ls = yh.std(ddof=1), lh.std()
                recent_y = ym[(ym.index >= t - 5) & (ym.index <= t - 1)]
                recent_l = h[(h["year"] >= t - 6) & (h["year"] <= t - 2)]["loc"]
                ya, la = max(ys, np.sqrt((recent_y ** 2).mean())), max(ls, np.sqrt((recent_l ** 2).mean()))
                g["a"] = [area.get((d, t), np.nan) for d in g["district"]]
                for mode, ysd, lsd in (("static", ys, ls), ("adaptive", ya, la)):
                    sd = np.sqrt(g["wsd"] ** 2 + ysd ** 2 + lsd ** 2)
                    lo, hi = g[0.5] - Z * sd, g[0.5] + Z * sd
                    rows.append(dict(crop=crop, lead=lead, year=t, level="district", mode=mode,
                                     inside=float(((g["obs"] >= lo) & (g["obs"] <= hi)).mean()),
                                     iscore=float(iscore(lo, hi, g["obs"]).mean()), n=len(g)))
                    ga = g.dropna(subset=["a"])
                    ga = ga[ga["a"] > 0]
                    for level, keys in (("national", np.array(["DE"] * len(ga))), ("state", ga["district"].str[:2].values)):
                        for k in np.unique(keys):
                            s = ga[keys == k]
                            if len(s) < 3:
                                continue
                            wt = s["a"] / s["a"].sum()
                            med, obs = float((wt * s[0.5]).sum()), float((wt * s["obs"]).sum())
                            sdn = np.sqrt(float((wt * s["wsd"]).sum()) ** 2 + ysd ** 2 + lsd ** 2 * float((wt ** 2).sum()))
                            lo_, hi_ = med - Z * sdn, med + Z * sdn
                            rows.append(dict(crop=crop, lead=lead, year=t, level=level, mode=mode,
                                             inside=float(lo_ <= obs <= hi_), iscore=float(iscore(lo_, hi_, obs)), n=1))
        print(crop, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(P, "ranges_adaptive_train.csv"), index=False)
    s = r.groupby(["crop", "level", "mode"]).apply(lambda g: pd.Series({
        "coverage80": np.average(g["inside"], weights=g["n"]), "iscore": np.average(g["iscore"], weights=g["n"])}),
        include_groups=False).unstack("mode").round(3)
    with pd.option_context("display.width", 200):
        print(s.to_string())


if __name__ == "__main__":
    main()
