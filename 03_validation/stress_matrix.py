"""v10 priority 3: stress-test matrix (training years only, forward errors).

Errors: forward predictions of the v8 model (fitted <= 2002 for 2003-2008, fitted <= 2008 for
2009-2017): bias = predicted - official (t/ha and %), RMSE.
Each district-year is classed by the weather in its sensitive window (district's flowering for maize
and potato, heading for wheat and barley; observed date, else the district median):
  water   climatic water balance (rain - ET0) from 10 days before to 20 days after the stage:
          dry / normal / wet = lower / middle / upper third of all training district-years 1979-2017
  heat    days with Tmax >= 30 C in the same window: hot = 3 or more
Also a whole-season class for maize (June-August water balance) because silage and grain maize
respond to late soil water (Peichl et al. 2018).
Writes data/processed/stress_matrix.csv.
"""
import os

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
STAGE = {"winter_wheat": "doy_heading", "winter_barley": "doy_heading", "grain_maize": "doy_flowering",
         "silage_maize": "doy_flowering", "potato": "doy_flowering"}


def windows(w, codes, df, stage, before=10, after=20):
    dates = pd.to_datetime(w["dates"])
    col = {c: i for i, c in enumerate(codes)}
    wb, hot = [], []
    for d, yr, s in zip(df["district"], df["year"], df[stage]):
        j = col.get(d)
        if j is None or not np.isfinite(s):
            wb.append(np.nan); hot.append(np.nan); continue
        t0 = np.datetime64(f"{yr}-01-01") + np.timedelta64(int(s) - 1 - before, "D")
        i0 = int((t0 - w["dates"][0]).astype(int))
        sl = slice(i0, i0 + before + after)
        wb.append(float(w["pr"][sl, j].sum() - w["et0"][sl, j].sum()))
        hot.append(float((w["tasmax"][sl, j] >= 30).sum()))
    return np.array(wb), np.array(hot)


def season_wb(w, codes, df):
    col = {c: i for i, c in enumerate(codes)}
    out = []
    for d, yr in zip(df["district"], df["year"]):
        j = col.get(d)
        if j is None:
            out.append(np.nan); continue
        i0 = int((np.datetime64(f"{yr}-06-01") - w["dates"][0]).astype(int))
        out.append(float(w["pr"][i0:i0 + 92, j].sum() - w["et0"][i0:i0 + 92, j].sum()))
    return np.array(out)


def main():
    w = dict(np.load(os.path.join(P, "weather_daily_v8.npz"), allow_pickle=True))   # into memory once
    codes = list(w["codes"])
    rows = []
    for crop, stage in STAGE.items():
        m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
        if stage not in m.columns:
            stage = "doy_heading" if "doy_heading" in m.columns else stage
        m = m[m["year"] <= 2017].copy()
        m[stage] = m[stage].fillna(m.groupby("district")[stage].transform("median")).fillna(m[stage].median())
        f02 = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2002.csv"), dtype={"district": str})
        f08 = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2008.csv"), dtype={"district": str})
        f = pd.concat([f02[f02["year"] <= 2008], f08[f08["year"] <= 2017]]).dropna(subset=["yield_t_ha", "pred"])
        # thresholds from all training district-years
        m["wb"], m["hot"] = windows(w, codes, m, stage)
        lo, hi = np.nanpercentile(m["wb"], [33.3, 66.7])
        f = f.merge(m[["district", "year", "wb", "hot"]], on=["district", "year"], how="left")
        f["water"] = np.select([f["wb"] < lo, f["wb"] > hi], ["dry", "wet"], "normal")
        f["heat"] = np.where(f["hot"] >= 3, "hot", "not hot")
        if "maize" in crop:
            m["swb"] = season_wb(w, codes, m)
            slo, shi = np.nanpercentile(m["swb"], [33.3, 66.7])
            f = f.merge(m[["district", "year", "swb"]], on=["district", "year"], how="left")
            f["season"] = np.select([f["swb"] < slo, f["swb"] > shi], ["dry summer", "wet summer"], "normal summer")
        f["err"] = f["pred"] - f["yield_t_ha"]
        groups = [("water x heat", ["water", "heat"]), ("water", ["water"])]
        if "maize" in crop:
            groups.append(("summer", ["season"]))
        for name, keys in groups:
            for k, g in f.groupby(keys):
                k = k if isinstance(k, tuple) else (k,)
                rows.append(dict(crop=crop, table=name, cell=" / ".join(k), n=len(g), years=g["year"].nunique(),
                                 bias=g["err"].mean(), bias_pct=100 * g["err"].mean() / g["yield_t_ha"].mean(),
                                 rmse=np.sqrt((g["err"] ** 2).mean()), obs_mean=g["yield_t_ha"].mean()))
        rows.append(dict(crop=crop, table="all", cell="all", n=len(f), years=f["year"].nunique(),
                         bias=f["err"].mean(), bias_pct=100 * f["err"].mean() / f["yield_t_ha"].mean(),
                         rmse=np.sqrt((f["err"] ** 2).mean()), obs_mean=f["yield_t_ha"].mean()))
        print(f"{crop}: window {stage}, dry < {lo:.0f} mm, wet > {hi:.0f} mm", flush=True)
    s = pd.DataFrame(rows).round(3)
    s.to_csv(os.path.join(P, "stress_matrix.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
