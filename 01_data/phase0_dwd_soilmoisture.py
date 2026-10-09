"""DWD crop-specific soil moisture (AMBAV 2.0 v1.5), district daily means 1991-2026 (v10, owner-approved).

Source: DWD Climate Data Center, grids_germany/daily/soil_moisture/<crop>/<year>/
  grids_germany_daily_soil_moisture_<crop>_<year>_0-60_v1.nc  (1 km, % of plant-available water = % nFK,
  soil from BUEK 1000 N, hourly station weather; updated on the 3rd of each month)
Crops: wheat (used for winter wheat and barley) and maize (grain / silage maize, potato).
Layer 0-60 cm (main root zone). Each file is downloaded, averaged per district (VG250, all valid cells:
the model already assumes the crop's soil profile), and deleted. Resumable per crop-year.
Writes data/processed/soil_moisture/sm_<crop>_<year>.npz (dates, codes, sm[days, districts]).
Run `python phase0_dwd_soilmoisture.py [first_year last_year]`.
"""
import os
import subprocess
import sys
import time

import netCDF4
import numpy as np
from pyproj import Transformer

from weather_districts import district_grid, read_means

BASE = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/soil_moisture"
P = os.path.join("data", "processed")
OUT = os.path.join(P, "soil_moisture")
TMP = os.path.join("data", "raw", "soil_moisture_tmp")
CURL = "curl"


def fetch(url, path, attempts=6):
    for attempt in range(attempts):
        r = subprocess.run([CURL, "-s", "-S", "-f", "-o", path + ".part", url])
        if r.returncode == 0:
            os.replace(path + ".part", path)
            return path
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def grid_index(nc, codes):
    """District index on the 1 km Gauss-Krueger grid, mapped to the weather file's district order."""
    cache = os.path.join(P, "district_grid_dwd_sm.npz")
    xs = [v for v in nc.variables if v.lower() in ("x", "easting", "rlon")]
    ys = [v for v in nc.variables if v.lower() in ("y", "northing", "rlat")]
    x, y = nc.variables[xs[0]][:], nc.variables[ys[0]][:]
    xx, yy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(31467, 4326, always_xy=True).transform(xx, yy)
    index, gcodes, _ = district_grid(lat, lon, cache=cache)
    pos = {c: i for i, c in enumerate(codes)}
    gmap = np.array([pos.get(c, -1) for c in gcodes])
    return np.where(index >= 0, gmap[np.clip(index, 0, None)], -1)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    codes = list(np.load(os.path.join(P, "weather_daily_v8.npz"))["codes"])
    years = range(int(sys.argv[1]), int(sys.argv[2]) + 1) if len(sys.argv) > 2 else range(1991, 2027)
    index = None
    for year in years:
        for crop in ("wheat", "maize"):
            out = os.path.join(OUT, f"sm_{crop}_{year}.npz")
            if os.path.exists(out):
                continue
            t0 = time.time()
            name = f"grids_germany_daily_soil_moisture_{crop}_{year}_0-60_v1.nc"
            path = os.path.join(TMP, name)
            try:
                fetch(f"{BASE}/{crop}/{year}/{name}", path)
            except RuntimeError as e:
                print(f"{crop} {year}: not available ({e})", flush=True)
                continue
            try:
                nc = netCDF4.Dataset(path)
                var = [v for v in nc.variables.values() if v.ndim == 3][0]
                var.set_auto_mask(True)
                if index is None:
                    index = grid_index(nc, codes)
                    print(f"grid {index.shape}, variable '{var.name}' ({getattr(var, 'units', '?')}), "
                          f"{np.sum(index >= 0)} cells in districts", flush=True)
                tvar = nc.variables["time"]
                dates = netCDF4.num2date(tvar[:], tvar.units, only_use_cftime_datetimes=False)
                dates = np.array([np.datetime64(str(d)[:10]) for d in dates])
                sm = read_means(var, index, len(codes))
                nc.close()
            finally:
                os.remove(path)
            np.savez_compressed(out, dates=dates, codes=np.array(codes), sm=sm.astype(np.float32))
            print(f"{crop} {year}: {len(dates)} days, mean {np.nanmean(sm):.0f} % nFK, districts with data "
                  f"{np.mean(np.isfinite(sm).any(0)) * 100:.0f} % ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
