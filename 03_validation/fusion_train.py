"""v11 A: fusion weights with MARS, national level, TRAINING years 2009-2017 only (docs/v11_plan.md).

Ours: in-season forecasts of 2009-2017 from the v9 forward models (fitted <= 2008; forecast_<crop>_v9_fwd2008),
national = area-weighted district medians, matched to each MARS bulletin as in phase5_benchmarks (latest
of our forecast dates on or before the bulletin's information date). Truth: official national yield.
Per crop and MARS month: errors e_o (ours), e_m (MARS); weight on ours
  w* = (var_m - cov) / (var_o + var_m - 2 cov), clipped to [0, 1], shrunk halfway to 0.5 (few years).
Rule: fusion adopted for a crop-month if it lowers RMSE in BOTH halves (2009-2012, 2013-2017) with weights
estimated on the OTHER half (cross-fitting), and overall. Writes data/processed/fusion_train.csv.
"""
import os

import numpy as np
import pandas as pd

import phase5_benchmarks as pb

B = os.path.join("data", "processed", "benchmarks")


def weight(eo, em, shrink=0.5):
    vo, vm, cov = np.var(eo), np.var(em), np.cov(eo, em, bias=True)[0, 1]
    den = vo + vm - 2 * cov
    w = 0.5 if den <= 1e-9 else float(np.clip((vm - cov) / den, 0, 1))
    return (1 - shrink) * w + shrink * 0.5


def main(crops=("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato")):
    t = pb.truth()
    off = pb.official(t)
    mars = off[off["source"].str.startswith("mars") & off["year"].between(2009, 2017) & (off["region"] == "DE")]
    rows = []
    for crop in crops:
        if not os.path.exists(os.path.join(pb.P, "forecast", f"forecast_{crop}_v9_fwd2008_lead00.csv")):
            continue
        o = pb.ours(crop, "_v9_fwd2008")
        o = o[(o["region"] == "DE") & (o["source"] == "ours_v9_fwd2008_blend")]
        pairs = []
        for _, r in mars[mars["crop"] == crop].iterrows():
            c = o[(o["year"] == r["year"]) & (o["date"] <= r["date"])]
            if not len(c):
                continue
            tr = t["yield_t_ha"].get((crop, "DE", r["year"]), np.nan)
            mo = int(r["source"][-2:])
            phase = "early (Mar-Jun)" if mo <= 6 else "mid (Jul)" if mo == 7 else "late (Aug-Oct)"
            pairs.append(dict(year=r["year"], month=phase, ours=c.sort_values("date").iloc[-1]["value"],
                              mars=r["value"], truth=tr))
        p = pd.DataFrame(pairs).dropna()
        p = p.groupby(["year", "month"], as_index=False).last()        # one value per year and phase
        for month, g in p.groupby("month"):
            if g["year"].nunique() < 6:
                continue
            g = g.assign(eo=g["ours"] - g["truth"], em=g["mars"] - g["truth"])
            h1, h2 = g[g["year"] <= 2012], g[g["year"] >= 2013]
            res, res25 = {}, {}
            for name, test, train in (("A", h1, h2), ("B", h2, h1)):
                w = weight(train["eo"].values, train["em"].values)
                fused = w * test["eo"] + (1 - w) * test["em"]
                res[name] = (np.sqrt((test["eo"] ** 2).mean()), np.sqrt((test["em"] ** 2).mean()), np.sqrt((fused ** 2).mean()), w)
                w25 = weight(train["eo"].values, train["em"].values, 0.25)
                f25 = w25 * test["eo"] + (1 - w25) * test["em"]
                res25[name] = np.sqrt((f25 ** 2).mean())
            w_all = weight(g["eo"].values, g["em"].values)
            rows.append(dict(crop=crop, month=month, years=g["year"].nunique(), w_ours=round(w_all, 2),
                             rmse_ours_A=res["A"][0], rmse_mars_A=res["A"][1], rmse_fused_A=res["A"][2],
                             rmse_ours_B=res["B"][0], rmse_mars_B=res["B"][1], rmse_fused_B=res["B"][2],
                             adopt=(res["A"][2] < res["A"][0]) and (res["B"][2] < res["B"][0]),
                             rmse_fused25_A=res25["A"], rmse_fused25_B=res25["B"],
                             w_ours25=round(weight(g["eo"].values, g["em"].values, 0.25), 2),
                             adopt25=(res25["A"] < res["A"][2]) and (res25["B"] < res["B"][2])
                             and (res25["A"] < res["A"][0]) and (res25["B"] < res["B"][0])))
    s = pd.DataFrame(rows).round(3)
    path = os.path.join(pb.P, "fusion_train.csv")            # keep rows of crops not run this time
    if os.path.exists(path) and os.path.getsize(path) > 5:
        old = pd.read_csv(path)
        if len(old) and "crop" in old.columns:
            s = pd.concat([old[~old["crop"].isin(crops)], s], ignore_index=True)
    s.to_csv(path, index=False)
    with pd.option_context("display.width", 220):
        print(s.to_string(index=False))


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"))
