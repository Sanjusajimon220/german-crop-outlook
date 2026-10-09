"""District weather v8: SARAH-3 satellite sunlight for 2021 onwards (district model improvement).

Our district sunlight (weather_daily.npz) comes from the HYRAS radiation grid up to 2020 and, from
2021, from an estimate based on station sunshine hours (Angstrom; 4 % low, noisier - checked against
pyranometers in phase0_solar_check.py). SARAH-3 matches the pyranometers within 1 %.
Here, for every district:
  SARAH district mean per day (0.05 deg cells inside the district, data/processed/weather/sarah3_germany.npz)
  level factor = mean HYRAS radiation / mean SARAH radiation over 2016-2020 (the level the models
                 were calibrated on is kept; only the day-to-day pattern changes)
  from 2021-01-01: rsds = SARAH x level factor; ET0 recomputed (FAO-56) for those days
Writes data/processed/weather_daily_v8.npz (weather_daily.npz = v7 stays unchanged) and prints the
pyranometer check for 2021+ (old estimate vs new). Use with environment variable VISTA_WEATHER=v8.
"""
import os

import numpy as np
import pandas as pd

from phase0_master import fao56_et0
from phase2_phenology import district_latitudes
from weather_districts import district_grid

P = os.path.join("data", "processed")


def main():
    w = dict(np.load(os.path.join(P, "weather_daily.npz")))
    codes = list(w["codes"])
    sar = np.load(os.path.join(P, "weather", "sarah3_germany.npz"))
    lat2, lon2 = np.meshgrid(sar["lat"].astype(float), sar["lon"].astype(float), indexing="ij")
    index, gcodes, _ = district_grid(lat2, lon2, cache=os.path.join(P, "district_grid_sarah.npz"))
    pos = {c: i for i, c in enumerate(codes)}
    gmap = np.array([pos.get(c, -1) for c in gcodes])
    cell_d = np.where(index.ravel() >= 0, gmap[np.clip(index.ravel(), 0, None)], -1)
    ok = cell_d >= 0
    flat = sar["sis"].reshape(len(sar["dates"]), -1)[:, ok].astype(np.float32)
    cnt = np.bincount(cell_d[ok], minlength=len(codes))
    sd = np.full((len(sar["dates"]), len(codes)), np.nan, np.float32)
    for t in range(len(sar["dates"])):
        v = flat[t]
        g = np.isfinite(v)
        s = np.bincount(cell_d[ok][g], weights=v[g], minlength=len(codes))
        n = np.bincount(cell_d[ok][g], minlength=len(codes))
        sd[t] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    print(f"SARAH district means: {np.sum(cnt > 0)} of {len(codes)} districts have cells", flush=True)
    dates = w["dates"]
    common, iw, isar = np.intersect1d(dates, sar["dates"], return_indices=True)
    ref = (common >= np.datetime64("2016-01-01")) & (common <= np.datetime64("2020-12-31"))
    factor = np.nanmean(w["rsds"][iw[ref]], 0) / np.nanmean(sd[isar[ref]], 0)
    print(f"level factor HYRAS / SARAH 2016-2020: median {np.nanmedian(factor):.3f} "
          f"(10-90 % {np.nanpercentile(factor, 10):.3f}-{np.nanpercentile(factor, 90):.3f})")
    new = (common >= np.datetime64("2021-01-01"))
    rs_new = sd[isar[new]] * factor[None, :]
    target = iw[new]
    old = w["rsds"][target].copy()
    good = np.isfinite(rs_new)
    w["rsds"][target] = np.where(good, rs_new, old)
    lat = district_latitudes()
    latv = np.array([lat.get(c, 51.0) for c in codes])
    sub = {k: w[k][target] for k in ("tas", "tasmax", "tasmin", "hurs", "rsds", "wind2")}
    w["et0"][target] = fao56_et0(sub, dates[target], latv).astype(np.float32)
    w["rsds_source"] = np.where(dates >= np.datetime64("2021-01-01"), "SARAH-3 (scaled)", "HYRAS")
    np.savez(os.path.join(P, "weather_daily_v8.npz"), **w)
    print(f"replaced {good.mean() * 100:.1f} % of district-days 2021+; mean radiation 2021+ old "
          f"{np.nanmean(old):.1f} -> new {np.nanmean(w['rsds'][target]):.1f} W/m2")
    # pyranometer check 2021+ (station -> its district)
    chk = pd.read_csv(os.path.join(P, "weather", "solar_station_check.csv"), parse_dates=["date"])
    chk = chk[chk["date"] >= "2021-01-01"]
    return w, chk


if __name__ == "__main__":
    main()
