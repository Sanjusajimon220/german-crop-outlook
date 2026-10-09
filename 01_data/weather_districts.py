"""Phase 0: daily district weather for Germany from the DWD HYRAS grids.

For each year: download the six HYRAS variables (1 km grids; radiation 5 km),
average every day over each district, save data/processed/weather/weather_<year>.npz,
and delete the raw grids again (they are ~400 MB per year; the result is ~2 MB).

Variables saved, each [days, districts]:
  tas, tasmax, tasmin  air temperature mean / max / min (deg C)
  hurs                 relative humidity (%)
  pr                   precipitation (mm/day)
  rsds                 global radiation (W/m2, daily mean); HYRAS has it only up to 2020,
                       later years get NaN here until a second source is added

  python weather_districts.py 2020          one year (keeps the raw files if already present)
  python weather_districts.py 1999 2025     a range of years
"""
import os
import subprocess
import sys
import time
import zipfile

import netCDF4
import numpy as np
import shapefile

RAW = os.path.join("data", "raw", "hyras_tmp")
OUT = os.path.join("data", "processed", "weather")
BOUNDARIES = os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
MASK = os.path.join("data", "processed", "district_grid.npz")
BASE = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/"
VARIABLES = {"tas": "air_temperature_mean", "tasmax": "air_temperature_max",
             "tasmin": "air_temperature_min", "hurs": "humidity", "pr": "precipitation"}
CURL = r"C:\Windows\System32\curl.exe"   # uses Windows' certificate store


def utm32(lat, lon):
    """Geographic coordinates (GRS80) -> UTM zone 32N metres, the projection of the
    district boundaries (EPSG:25832). Standard transverse Mercator series (Snyder 1987)."""
    a, f, k0, lon0 = 6378137.0, 1 / 298.257222101, 0.9996, np.radians(9.0)
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = np.radians(lat), np.radians(lon)
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    t, c = np.tan(phi) ** 2, ep2 * np.cos(phi) ** 2
    A = np.cos(phi) * (lam - lon0)
    m = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))
    x = k0 * n * (A + (1 - t + c) * A ** 3 / 6 + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * A ** 5 / 120)
    y = k0 * (m + n * np.tan(phi) * (A ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * A ** 4 / 24
                                     + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2) * A ** 6 / 720))
    return x + 500000.0, y


def inside(px, py, ring):
    """Even-odd ray casting: which points lie inside one polygon ring [k, 2]."""
    result = np.zeros(len(px), bool)
    x0, y0 = ring[:-1, 0], ring[:-1, 1]
    x1, y1 = ring[1:, 0], ring[1:, 1]
    for a, b, c, d in zip(x0, y0, x1, y1):
        crosses = (b > py) != (d > py)
        if crosses.any():
            xi = a + (py[crosses] - b) * (c - a) / (d - b)
            result[np.flatnonzero(crosses)[px[crosses] < xi]] ^= True
    return result


def district_grid(lat, lon, cache=MASK):
    """District index (or -1) for every cell of a grid with given lat/lon, cached on disk."""
    if os.path.exists(cache):
        d = np.load(cache)
        return d["index"], d["codes"], d["names"]
    with zipfile.ZipFile(BOUNDARIES) as z:
        stem = "vg250_ebenen_1231/VG250_KRS"
        sf = shapefile.Reader(shp=z.open(stem + ".shp"), shx=z.open(stem + ".shx"),
                              dbf=z.open(stem + ".dbf"), encoding="utf-8")
        records, shapes = sf.records(), sf.shapes()
    gx, gy = utm32(lat.ravel(), lon.ravel())
    index = np.full(gx.size, -1, np.int16)
    codes, names = [], []
    for rec, shp in zip(records, shapes):
        rec = rec.as_dict()
        if rec.get("GF") != 4:              # 4 = land area with structure; skips water parts
            continue
        k = len(codes)
        codes.append(rec["AGS"])
        names.append(rec["GEN"])
        x0, y0, x1, y1 = shp.bbox
        box = np.flatnonzero((gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1))
        pts = np.array(shp.points)
        hit = np.zeros(len(box), bool)
        for start, end in zip(shp.parts, list(shp.parts[1:]) + [len(pts)]):
            hit ^= inside(gx[box], gy[box], pts[start:end])   # holes cancel out
        index[box[hit]] = k
    index = index.reshape(lat.shape)
    np.savez(cache, index=index, codes=np.array(codes), names=np.array(names))
    return index, np.array(codes), np.array(names)


def fetch(var_dir, name, attempts=6):
    """Download one file; the DWD server sometimes drops connections, so retry with pauses.
    A partial download is written under a temporary name, so it is never mistaken for a file."""
    path = os.path.join(RAW, name)
    if os.path.exists(path):
        return path
    for attempt in range(attempts):
        result = subprocess.run([CURL, "-s", "-S", "-f", "-o", path + ".part", BASE + var_dir + "/" + name])
        if result.returncode == 0:
            os.replace(path + ".part", path)
            return path
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed {attempts} times: {name}")


def read_means(variable, index, n, rows=None, cols=None, chunk=31):
    """District means of a NetCDF variable [days, rows, cols], read a month at a time to keep
    memory low (a whole year unpacked would need ~1.6 GB). rows/cols pick coarse cells."""
    parts = []
    for start in range(0, variable.shape[0], chunk):
        grid = variable[start:start + chunk].filled(np.nan).astype(np.float32)
        if rows is not None:
            grid = grid[:, rows][:, :, cols]
        parts.append(district_means(grid, index, n))
    return np.concatenate(parts)


def district_means(grid, index, n):
    """Daily grid [days, rows, cols] -> daily district means [days, n]."""
    flat = grid.reshape(grid.shape[0], -1)
    idx = index.ravel()
    ok = idx >= 0
    out = np.empty((grid.shape[0], n), np.float32)
    count = np.bincount(idx[ok], minlength=n)
    for t in range(grid.shape[0]):
        v = flat[t, ok]
        good = np.isfinite(v)
        out[t] = (np.bincount(idx[ok][good], weights=v[good], minlength=n)
                  / np.maximum(np.bincount(idx[ok][good], minlength=n), 1))
    out[:, count == 0] = np.nan
    return out


def process_year(year, keep=False):
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    out, paths = {}, []
    index = codes = None
    for var, var_dir in VARIABLES.items():
        path = fetch(var_dir, f"{var}_hyras_1_{year}_v6-1_de.nc")
        paths.append(path)
        with netCDF4.Dataset(path) as d:
            if index is None:
                index, codes, names = district_grid(d["lat"][:], d["lon"][:])
                x, y = d["x"][:], d["y"][:]
                time = netCDF4.num2date(d["time"][:], d["time"].units)
            out[var] = read_means(d[var], index, len(codes))
    # radiation: 5 km grid in the same projection; each 1 km cell takes the 5 km cell it lies in
    if year <= 2020:
        path = fetch("radiation_global", f"rsds_hyras_5_{year}_v4-0_de.nc")
        paths.append(path)
        with netCDF4.Dataset(path) as d:
            cx, cy = d["x"][:], d["y"][:]
            col = np.clip(np.round((x - cx[0]) / (cx[1] - cx[0])).astype(int), 0, len(cx) - 1)
            row = np.clip(np.round((y - cy[0]) / (cy[1] - cy[0])).astype(int), 0, len(cy) - 1)
            out["rsds"] = read_means(d["rsds"], index, len(codes), row, col)
    else:
        out["rsds"] = np.full_like(out["tas"], np.nan)
    dates = np.array([np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}") for t in time])
    np.savez(os.path.join(OUT, f"weather_{year}.npz"), dates=dates, codes=codes, **out)
    if not keep:
        for p in paths:
            os.remove(p)
    return out, codes


def main():
    years = [int(a) for a in sys.argv[1:]] or [2020]
    years = list(range(years[0], years[-1] + 1))
    for year in years:
        if os.path.exists(os.path.join(OUT, f"weather_{year}.npz")):
            print(f"{year}: already processed")
            continue
        keep = len(years) == 1        # a single test year keeps its raw files
        out, codes = process_year(year, keep)
        print(f"{year}: {len(codes)} districts, mean temperature {np.nanmean(out['tas']):.1f} C, "
              f"rain {np.nanmean(out['pr'].sum(0)):.0f} mm, "
              f"radiation {np.nanmean(out['rsds']):.0f} W/m2", flush=True)


if __name__ == "__main__":
    main()
