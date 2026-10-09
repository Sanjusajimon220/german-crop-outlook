"""v10 combined forward test, evaluation (training years only: forward predictions 2009-2017, model
fitted <= 2008, v8 weather).

Per crop and variant file forward_predictions_<crop>_v8<variant>_cut2008.csv:
  rmse, bias; stress gap (bias dry+hot flowering window - bias normal, as eval_step3.py);
  RMSE in the drought years of the period (district-years with summer water balance in the driest
  third); year-wide and state-year mean-square shares;
  + offsets: the same after district offsets learned from the current model's 2003-2008 forward errors
  (year part removed, shrinkage k = 3; test_district_offsets.py). Approximation: offsets of each variant
  would strictly come from its own 2003-2008 errors (to be redone for the finalists with cut 2002).
Writes data/processed/combined_forward_eval.csv.
"""
import glob
import os

import numpy as np
import pandas as pd

from stress_matrix import STAGE, season_wb, windows

P = os.path.join("data", "processed")
K = 3


def offsets(crop):
    a = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2002.csv"), dtype={"district": str})
    a = a[a["year"] <= 2008].dropna(subset=["yield_t_ha", "pred"]).copy()
    a["e"] = a["pred"] - a["yield_t_ha"]
    a["e"] -= a.groupby("year")["e"].transform("mean")
    g = a.groupby("district")["e"]
    return g.sum() / (g.count() + K)


def main():
    w = dict(np.load(os.path.join(P, "weather_daily_v8.npz"), allow_pickle=True))
    codes = list(w["codes"])
    rows = []
    for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
        m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
        m = m[(m["year"] >= 2009) & (m["year"] <= 2017)].copy()
        st = STAGE[crop] if STAGE[crop] in m.columns else "doy_heading"
        m[st] = m[st].fillna(m.groupby("district")[st].transform("median")).fillna(m[st].median())
        m["wb"], m["hot"] = windows(w, codes, m, st)
        m["swb"] = season_wb(w, codes, m)
        lo, hi = np.nanpercentile(m["wb"], [33.3, 66.7])
        slo = np.nanpercentile(m["swb"], 33.3)
        off = offsets(crop)
        for f in sorted(glob.glob(os.path.join(P, f"forward_predictions_{crop}_v8*_cut2008.csv"))):
            name = os.path.basename(f)[len(f"forward_predictions_{crop}_v8"):-len("_cut2008.csv")].strip("_") or "current"
            p = pd.read_csv(f, dtype={"district": str}).dropna(subset=["yield_t_ha", "pred"])
            p = p.merge(m[["district", "year", "wb", "hot", "swb"]], on=["district", "year"], how="left")
            for tag, pred in (("", p["pred"]), (" + offsets", p["pred"] - p["district"].map(off).fillna(0))):
                e = pred - p["yield_t_ha"]
                dry, wet, hot = p["wb"] < lo, p["wb"] > hi, p["hot"] >= 3
                normal = ~dry & ~wet & ~hot
                yr = e.groupby(p["year"]).transform("mean")
                sy = e.groupby([p["year"], p["district"].str[:2]]).transform("mean") - yr
                tot = (e ** 2).sum()
                rows.append(dict(crop=crop, variant=name + tag, n=len(p), rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
                                 stress_gap=e[dry & hot].mean() - e[normal].mean(),
                                 rmse_dry_summer=np.sqrt((e[p["swb"] < slo] ** 2).mean()),
                                 share_year=(yr ** 2).sum() / tot, share_state_year=(sy ** 2).sum() / tot))
    s = pd.DataFrame(rows).round(3)
    s.to_csv(os.path.join(P, "combined_forward_eval.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
