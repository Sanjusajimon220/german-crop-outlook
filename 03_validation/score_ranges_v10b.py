"""v10 priority 2: the ONE test look at the v10 ranges (test years 2018-2025), against the v9 ranges.

District: 80 % coverage, interval score (width + 10 x miss outside the range; lower = better; Gneiting &
Raftery 2007), share of PIT in the outer 10 % tails (ideal 0.20), Brier score of P(below normal)
(event = official yield < district's previous 5-year official mean) against the base rate (share of
such events in the forward years 2009-2017).
State / Germany: the same against the official final yields (truth_official.csv); v9 has no
aggregate ranges (districts drawn independently would give near-zero width), so only v10 is shown.
Writes data/processed/ranges_v10/score_v10b.csv.
"""
import os

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
D = os.path.join(P, "ranges_v10b")
CROPS = ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato")
LEADS = (12, 8, 6, 4, 2, 0)


def iscore(lo, hi, y, alpha=0.2):
    return (hi - lo) + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def base_rate(crop):
    name = f"forward_predictions_{crop}{'_potato_canopy' if crop == 'potato' else ''}_modis_cut2008.csv"
    f = pd.read_csv(os.path.join(P, name), dtype={"district": str}).dropna(subset=["yield_t_ha"])
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    h = y[y["crop"] == crop].set_index(["district", "year"])["yield_t_ha"]
    thr = np.array([np.nanmean([h.get((d, yr - k), np.nan) for k in range(1, 6)]) for d, yr in zip(f["district"], f["year"])])
    ok = np.isfinite(thr)
    return float((f["yield_t_ha"].values[ok] < thr[ok]).mean())


def main():
    truth = pd.read_csv(os.path.join(P, "benchmarks", "truth_official.csv"), dtype={"region": str})
    truth = truth.set_index(["crop", "region", "year"])["yield_t_ha"]
    rows = []
    for crop in CROPS:
        base = base_rate(crop)
        for lead in LEADS:
            r = pd.read_csv(os.path.join(D, f"ranges_{crop}_lead{lead:02d}.csv"), dtype={"region": str})
            agg = r["level"] != "district"
            r.loc[agg, "obs"] = [truth.get((crop, reg, yr), np.nan) for reg, yr in
                                 zip(r.loc[agg, "region"], r.loc[agg, "year"])]
            old = pd.read_csv(os.path.join(P, f"forecast_intervals_{crop}_v9.csv"), dtype={"district": str})
            old = old[old["lead_weeks"] == lead]
            for level in ("district", "state", "national"):
                g = r[(r["level"] == level) & r["obs"].notna()]
                ev = (g["obs"] < g["normal"]).astype(float)
                okb = g["normal"].notna()
                row = dict(crop=crop, lead=lead, level=level, n=len(g),
                           cov80=((g["obs"] >= g["low80"]) & (g["obs"] <= g["high80"])).mean(),
                           iscore=iscore(g["low80"], g["high80"], g["obs"]).mean(),
                           width=(g["high80"] - g["low80"]).mean(),
                           brier=((g["p_below"] - ev)[okb] ** 2).mean(),
                           brier_base=((base - ev)[okb] ** 2).mean(), base_rate=base)
                if level == "district":
                    row["pit_tails"] = ((g["pit"] < 0.1) | (g["pit"] > 0.9)).mean()
                    o = old[old["obs"].notna()]
                    row["v9_cov80"] = ((o["obs"] >= o["low80"]) & (o["obs"] <= o["high80"])).mean()
                    row["v9_iscore"] = iscore(o["low80"], o["high80"], o["obs"]).mean()
                    row["v9_width"] = (o["high80"] - o["low80"]).mean()
                rows.append(row)
    s = pd.DataFrame(rows).round(3)
    s.to_csv(os.path.join(D, "score_v10b.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
