"""Field weather for any list of points (e.g. the YieldSAT fields): DWD HYRAS-DE 1 km daily grids.

Points: a CSV with columns point_id, lat, lon (default data/raw/yieldsat/wheat_metadata.csv:
field_shared_name, centroid_latitude_wgs84, centroid_longitude_wgs84). Each point gets its
nearest 1 km cell; per year the five variables (tas, tasmax, tasmin, hurs, pr) are downloaded,
only those cells kept, and the raw grids deleted again (same downloader as weather_districts.py).
Writes data/processed/yieldsat/hyras_points_<year>.npz (dates, point_ids, [days, points] per variable).
Run `python phase0_hyras_points.py 2015 2019`.
"""
import os
import sys

import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from weather_districts import RAW, VARIABLES, fetch, utm32

OUT = os.path.join("data", "processed", "yieldsat")


def points():
    m = pd.read_csv(os.path.join("data", "raw", "yieldsat", "wheat_metadata.csv"))
    return m["field_shared_name"].values, m["centroid_latitude_wgs84"].values, m["centroid_longitude_wgs84"].values


def process_year(year):
    ids, plat, plon = points()
    out, paths, cells = {}, [], None
    for var, var_dir in VARIABLES.items():
        path = fetch(var_dir, f"{var}_hyras_1_{year}_v6-1_de.nc")
        paths.append(path)
        with netCDF4.Dataset(path) as d:
            if cells is None:
                gx, gy = utm32(d["lat"][:].ravel(), d["lon"][:].ravel())
                px, py = utm32(plat, plon)
                dist, cells = cKDTree(np.column_stack([gx, gy])).query(np.column_stack([px, py]))
                print(f"  nearest cell distance: max {dist.max():.0f} m", flush=True)
            time = netCDF4.num2date(d["time"][:], d["time"].units)
            v = d[var]
            parts = []
            for a in range(0, v.shape[0], 31):
                g = v[a:a + 31].filled(np.nan).astype(np.float32)
                parts.append(g.reshape(g.shape[0], -1)[:, cells])
            out[var] = np.concatenate(parts)
    dates = np.array([np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}") for t in time])
    np.savez_compressed(os.path.join(OUT, f"hyras_points_{year}.npz"), dates=dates, point_ids=ids, **out)
    for p in paths:
        os.remove(p)
    print(f"{year}: {len(ids)} points, rain {np.nanmean(out['pr'].sum(0)):.0f} mm, "
          f"temperature {np.nanmean(out['tas']):.1f} C", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    for year in range(int(sys.argv[1]), int(sys.argv[-1]) + 1):
        if os.path.exists(os.path.join(OUT, f"hyras_points_{year}.npz")):
            print(f"{year}: already processed")
            continue
        process_year(year)


if __name__ == "__main__":
    main()
