"""District weather v8 extended to 2026 (for the pre-registered 2026 state check).

Base: weather_daily_v8.npz (1979-2025). Added 2026:
  temperature, rain, humidity  HYRAS-DE 2026 v6-1 district means (weather_districts.process_year)
  sunlight                     SARAH-3 district means x the same HYRAS/SARAH level factor as v8
  wind                         DWD station data (dwd_stations_recent, same interpolation as phase0_stations)
  days after the last HYRAS day  each variable = the district's mean for that calendar day 2016-2025
                                 (only matters for late maize; winter cereals are harvested by then)
  ET0                          FAO-56 from the above (as all other years)
Writes data/processed/weather_daily_v8_2026.npz (use with VISTA_WEATHER=v8_2026).
"""
import glob
import os

import numpy as np

from phase0_master import fao56_et0
import phase0_stations
from phase0_stations import RECENT, district_centres, idw, read_station, station_positions, MAX_KM
from phase2_phenology import district_latitudes
from weather_districts import district_grid

P = os.path.join("data", "processed")


def sarah_districts(codes):
    sar = np.load(os.path.join(P, "weather", "sarah3_germany.npz"))
    lat2, lon2 = np.meshgrid(sar["lat"].astype(float), sar["lon"].astype(float), indexing="ij")
    index, gcodes, _ = district_grid(lat2, lon2, cache=os.path.join(P, "district_grid_sarah.npz"))
    pos = {c: i for i, c in enumerate(codes)}
    gmap = np.array([pos.get(c, -1) for c in gcodes])
    cell = np.where(index.ravel() >= 0, gmap[np.clip(index.ravel(), 0, None)], -1)
    ok = cell >= 0
    flat = sar["sis"].reshape(len(sar["dates"]), -1)[:, ok].astype(np.float32)
    out = np.full((len(sar["dates"]), len(codes)), np.nan, np.float32)
    for t in range(len(sar["dates"])):
        g = np.isfinite(flat[t])
        s = np.bincount(cell[ok][g], weights=flat[t][g], minlength=len(codes))
        n = np.bincount(cell[ok][g], minlength=len(codes))
        out[t] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return sar["dates"], out


def main():
    w = dict(np.load(os.path.join(P, "weather_daily_v8.npz")))
    codes = list(w["codes"])
    h = np.load(os.path.join(P, "weather", "weather_2026.npz"))
    assert list(h["codes"]) == codes
    days = np.arange(np.datetime64("2026-01-01"), np.datetime64("2027-01-01"))
    n = len(days)
    new = {k: np.full((n, len(codes)), np.nan, np.float32) for k in ("tas", "tasmax", "tasmin", "hurs", "pr", "rsds", "wind2")}
    k = (h["dates"] - days[0]).astype(int)
    for v in ("tas", "tasmax", "tasmin", "hurs", "pr"):
        new[v][k] = h[v]
    # sunlight: SARAH with the v8 level factor (HYRAS / SARAH, 2016-2020)
    sd, sv = sarah_districts(codes)
    common, iw, isar = np.intersect1d(w["dates"], sd, return_indices=True)
    ref = (common >= np.datetime64("2016-01-01")) & (common <= np.datetime64("2020-12-31"))
    factor = np.nanmean(w["rsds"][iw[ref]], 0) / np.nanmean(sv[isar[ref]], 0)
    c2, i_new, i_sar = np.intersect1d(days, sd, return_indices=True)
    new["rsds"][i_new] = sv[i_sar] * factor[None, :]
    # wind: stations (recent files) interpolated as in phase0_stations
    pos = station_positions()
    phase0_stations.START, phase0_stations.END = days[0], days[-1]      # station reader on the 2026 axis
    rec = {}
    for path in sorted(glob.glob(os.path.join(RECENT, "*.zip"))):
        sid, wind, _ = read_station(path, days)
        if sid in pos and np.isfinite(wind).any():
            rec[sid] = wind
    ids = list(rec)
    winds = np.array([rec[s] for s in ids]).T
    dcodes, clat, clon = district_centres()
    assert list(dcodes) == codes
    slat = np.array([pos[s][0] for s in ids]); slon = np.array([pos[s][1] for s in ids])
    dist = np.hypot((clat[:, None] - slat[None, :]) * 111.2,
                    (clon[:, None] - slon[None, :]) * 111.2 * np.cos(np.radians(clat))[:, None])
    wd = idw(winds, dist)
    gap = np.isnan(wd)
    wd[gap] = idw(winds, dist, candidates=60, max_km=150.0)[gap]
    new["wind2"] = (wd * 4.87 / np.log(67.8 * 10 - 5.42)).astype(np.float32)
    # fill the remaining days with the district's calendar-day mean 2016-2025
    hist = (w["dates"] >= np.datetime64("2016-01-01"))
    doy_hist = (w["dates"][hist] - w["dates"][hist].astype("datetime64[Y]")).astype(int)
    doy_new = (days - days.astype("datetime64[Y]")).astype(int)
    filled = {}
    for v in new:
        clim = np.stack([np.nanmean(w[v][hist][doy_hist == d], 0) for d in range(366)])
        miss = np.isnan(new[v])
        filled[v] = miss.all(1).sum()
        new[v] = np.where(miss, clim[np.clip(doy_new, 0, 365)], new[v])
    lat = district_latitudes()
    new["et0"] = fao56_et0(new, days, np.array([lat.get(c, 51.0) for c in codes])).astype(np.float32)
    out = {}
    for key, val in w.items():
        if key in new:
            out[key] = np.concatenate([val, new[key]])
        elif key == "dates":
            out[key] = np.concatenate([val, days])
        elif isinstance(val, np.ndarray) and val.shape[:1] == w["dates"].shape and key != "codes":
            pad = np.full((n,) + val.shape[1:], val[-1] if val.dtype.kind in "USO" else False, dtype=val.dtype)
            out[key] = np.concatenate([val, pad])
        else:
            out[key] = val
    np.savez(os.path.join(P, "weather_daily_v8_2026.npz"), **out)
    last = h["dates"].max()
    print(f"2026 added: HYRAS to {last}, SARAH to {sd.max()}, wind stations {len(ids)}; days filled with "
          f"climatology: " + ", ".join(f"{v} {c}" for v, c in filled.items()))
    print(f"2026 Jan-Sep: mean temperature {np.nanmean(new['tas'][:273]):.1f} C, rain "
          f"{np.nanmean(new['pr'][:273].sum(0)):.0f} mm, ET0 {np.nanmean(new['et0'][:273].sum(0)):.0f} mm")


if __name__ == "__main__":
    main()
