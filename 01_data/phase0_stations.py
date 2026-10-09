"""Phase 0: daily wind and sunlight per district from DWD weather stations.

HYRAS has no wind, and its sunlight (global radiation) stops in 2020. The DWD daily
station records (data/raw/dwd_stations) have both wind speed and sunshine hours.

1. Read the stations' daily mean wind speed (FM, m/s at ~10 m) and sunshine duration (SDK, h).
2. Spread them to each district's centre by inverse-distance weighting: the up to 6 nearest
   stations with data that day within 75 km, each weighted by 1 / distance^2.
3. Convert sunshine hours to sunlight energy with the Angstrom formula
       radiation = top-of-atmosphere radiation x (a + b x sunshine hours / day length)
   with a and b fitted against the HYRAS radiation of 1979-2020.
4. Save data/processed/weather/stations.npz: dates, districts, wind_10m, sunshine_h,
   rsds_angstrom (W/m2), and print how well the Angstrom radiation matches HYRAS.

Run `python phase0_stations.py` after weather_districts.py and phase0_soil.py.
"""
import glob
import io
import os
import zipfile

import netCDF4
import numpy as np

RAW = os.path.join("data", "raw", "dwd_stations")
RECENT = os.path.join("data", "raw", "dwd_stations_recent")   # newest ~1.5 years
WEATHER = os.path.join("data", "processed", "weather")
GRID = os.path.join("data", "processed", "district_grid_soil.npz")
SOIL_NC = os.path.join("data", "raw", "soil", "AG_SOILINFO_THETAFC.nc")
START, END = np.datetime64("1979-01-01"), np.datetime64("2025-12-31")
MAX_KM, NEAREST = 75.0, 6


def station_positions():
    """Station id -> (lat, lon) from the fixed-width station description file."""
    pos = {}
    with open(os.path.join(RAW, "KL_Tageswerte_Beschreibung_Stationen.txt"), encoding="latin-1") as f:
        for line in f.readlines()[2:]:
            parts = line.split()
            if len(parts) > 5 and parts[0].isdigit():
                pos[int(parts[0])] = (float(parts[4]), float(parts[5]))
    return pos


def read_station(path, days):
    """One station zip -> (id, wind [days], sunshine [days]) on the common day axis."""
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if n.startswith("produkt_klima_tag"))
    lines = io.TextIOWrapper(z.open(name), encoding="latin-1").read().splitlines()
    header = [h.strip() for h in lines[0].split(";")]
    i_date, i_fm, i_sdk = header.index("MESS_DATUM"), header.index("FM"), header.index("SDK")
    wind = np.full(len(days), np.nan, np.float32)
    sun = np.full(len(days), np.nan, np.float32)
    sid = None
    for line in lines[1:]:
        c = line.split(";")
        d = c[i_date].strip()
        day = np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
        if not START <= day <= END:
            continue
        k = (day - START).astype(int)
        sid = int(c[0])
        fm, sdk = float(c[i_fm]), float(c[i_sdk])
        wind[k] = fm if fm >= 0 else np.nan          # -999 = missing
        sun[k] = sdk if sdk >= 0 else np.nan
    return sid, wind, sun


def district_centres():
    g = np.load(GRID)
    with netCDF4.Dataset(SOIL_NC) as d:
        lat, lon = d["lat"][:].ravel(), d["lon"][:].ravel()
    idx = g["index"].ravel()
    n = len(g["codes"])
    count = np.bincount(idx[idx >= 0], minlength=n)
    clat = np.bincount(idx[idx >= 0], weights=lat[idx >= 0], minlength=n) / count
    clon = np.bincount(idx[idx >= 0], weights=lon[idx >= 0], minlength=n) / count
    return g["codes"], clat, clon


def idw(values, dist_km, candidates=30, max_km=MAX_KM):
    """values [days, stations], dist_km [districts, stations] -> [days, districts].

    Each district looks at its `candidates` closest stations; on each day the first NEAREST
    of them that have data (and lie within MAX_KM) are averaged with weights 1 / distance^2.
    """
    order = np.argsort(dist_km, 1)[:, :candidates]                    # [districts, c]
    d = np.take_along_axis(dist_km, order, 1)
    weight = np.where(d <= max_km, 1.0 / np.maximum(d, 1.0) ** 2, 0.0)
    out = np.full((values.shape[0], dist_km.shape[0]), np.nan, np.float32)
    for t in range(values.shape[0]):
        v = values[t][order]                                           # [districts, c]
        ok = np.isfinite(v) & (weight > 0)
        ok &= np.cumsum(ok, 1) <= NEAREST                               # first NEAREST valid ones
        w = np.where(ok, weight, 0.0)
        total = w.sum(1)
        out[t] = np.where(total > 0, (w * np.nan_to_num(v)).sum(1) / np.maximum(total, 1e-12), np.nan)
    return out


def top_of_atmosphere(lat_deg, days):
    """Daily extraterrestrial radiation (W/m2) and day length (h), FAO-56 equations 21-34."""
    doy = (days - days.astype("datetime64[Y]")).astype(int) + 1
    phi = np.radians(lat_deg)[None, :]
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)[:, None]
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)[:, None]
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(decl)
                                            + np.cos(phi) * np.cos(decl) * np.sin(ws))  # MJ/m2/day
    return ra * 1e6 / 86400.0, 24.0 / np.pi * ws


def main():
    days = np.arange(START, END + np.timedelta64(1, "D"))
    pos = station_positions()
    records = {}
    # historical files first; the recent ones fill the days the historical ones lack
    for path in sorted(glob.glob(os.path.join(RAW, "*.zip"))) + sorted(glob.glob(os.path.join(RECENT, "*.zip"))):
        sid, wind, sun = read_station(path, days)
        if sid not in pos:
            continue
        if sid in records:
            old_wind, old_sun = records[sid]
            wind = np.where(np.isfinite(old_wind), old_wind, wind)
            sun = np.where(np.isfinite(old_sun), old_sun, sun)
        records[sid] = (wind, sun)
    ids = [s for s, (w_, s_) in records.items() if np.isfinite(w_).any() or np.isfinite(s_).any()]
    winds = np.array([records[s][0] for s in ids]).T                  # [days, stations]
    suns = np.array([records[s][1] for s in ids]).T
    print(f"{len(ids)} stations; days with wind at >= 1 station: "
          f"{np.isfinite(winds).any(1).mean():.0%}, with sunshine: {np.isfinite(suns).any(1).mean():.0%}")

    codes, clat, clon = district_centres()
    slat = np.array([pos[s][0] for s in ids])
    slon = np.array([pos[s][1] for s in ids])
    dy = (clat[:, None] - slat[None, :]) * 111.2
    dx = (clon[:, None] - slon[None, :]) * 111.2 * np.cos(np.radians(clat))[:, None]
    dist = np.hypot(dx, dy)
    wind_d, sun_d = idw(winds, dist), idw(suns, dist)
    # the few district-days with no station within MAX_KM fall back to stations up to 150 km away
    for arr, raw in ((wind_d, winds), (sun_d, suns)):
        gap = np.isnan(arr)
        if gap.any():
            arr[gap] = idw(raw, dist, candidates=60, max_km=150.0)[gap]

    ra, daylen = top_of_atmosphere(clat, days)
    rel = np.clip(sun_d / daylen, 0, 1)
    # fit a, b against HYRAS radiation where both exist (1999-2020)
    hyras = np.full_like(sun_d, np.nan)
    for year in range(1979, 2021):
        path = os.path.join(WEATHER, f"weather_{year}.npz")
        if os.path.exists(path):
            w = np.load(path)
            k = (w["dates"] - START).astype(int)
            hyras[k] = w["rsds"]
    ok = np.isfinite(hyras) & np.isfinite(rel) & (ra > 0)
    A = np.stack([ra[ok], ra[ok] * rel[ok]], 1)
    (a, b), *_ = np.linalg.lstsq(A, hyras[ok], rcond=None)
    rsds = (ra * (a + b * rel)).astype(np.float32)
    err = rsds[ok] - hyras[ok]
    print(f"Angstrom fit: a = {a:.3f}, b = {b:.3f} (FAO default 0.25, 0.50)")
    print(f"vs HYRAS radiation 1979-2020: daily RMSE {np.sqrt((err ** 2).mean()):.1f} W/m2, "
          f"bias {err.mean():+.1f} W/m2, correlation {np.corrcoef(rsds[ok], hyras[ok])[0, 1]:.3f}")
    last = days[np.isfinite(sun_d).all(1)][-1]
    print(f"sunshine complete for every district up to {last}")
    print(f"district wind at 10 m: mean {np.nanmean(wind_d):.2f} m/s; "
          f"missing district-days: wind {np.isnan(wind_d).mean():.1%}, sunshine {np.isnan(sun_d).mean():.1%}")
    np.savez(os.path.join(WEATHER, "stations.npz"), dates=days, codes=codes, wind_10m=wind_d,
             sunshine_h=sun_d, rsds_angstrom=rsds, angstrom=np.array([a, b]))


if __name__ == "__main__":
    main()
