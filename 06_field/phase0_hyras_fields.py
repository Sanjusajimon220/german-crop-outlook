"""Field weather: DWD HYRAS-DE 1 km daily grids for the Rur fields (2020-2026).

For each year the five 1 km variables (tas, tasmax, tasmin, hurs, pr) are downloaded from DWD
Open Data, only the grid cells that hold a field are kept, and the raw grids are deleted again
(same downloader as weather_districts.py). Every field (centroid, fields_district.csv) gets its
nearest 1 km cell.
Writes data/processed/field/hyras_cells_<year>.npz (dates, cell ids, [days, cells] per variable)
and data/processed/field/fields_hyras_cell.csv (field record -> cell).
Radiation and wind stay from the district weather for now (HYRAS radiation ends 2020; SARAH-3
satellite radiation planned). Run `python phase0_hyras_fields.py 2020 2026`.
"""
import os
import sys

import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from weather_districts import RAW, VARIABLES, fetch, utm32

FD = os.path.join("data", "processed", "field")


def cells_for_fields(lat, lon):
    f = pd.read_csv(os.path.join(FD, "fields_district.csv"))
    gx, gy = utm32(lat.ravel(), lon.ravel())
    box = np.flatnonzero((gx > f["x"].min() - 3000) & (gx < f["x"].max() + 3000) &
                         (gy > f["y"].min() - 3000) & (gy < f["y"].max() + 3000))
    _, k = cKDTree(np.column_stack([gx[box], gy[box]])).query(f[["x", "y"]].values)
    f["cell"] = box[k]                      # flat index into the HYRAS grid
    f[["ID", "VALIDFROM", "CODE", "cell"]].to_csv(os.path.join(FD, "fields_hyras_cell.csv"), index=False)
    return np.unique(f["cell"].values)


def process_year(year, cells=None):
    out, paths = {}, []
    for var, var_dir in VARIABLES.items():
        path = fetch(var_dir, f"{var}_hyras_1_{year}_v6-1_de.nc")
        paths.append(path)
        with netCDF4.Dataset(path) as d:
            if cells is None:
                cells = cells_for_fields(d["lat"][:], d["lon"][:])
            time = netCDF4.num2date(d["time"][:], d["time"].units)
            v = d[var]
            parts = []
            for a in range(0, v.shape[0], 31):          # a month at a time (memory)
                g = v[a:a + 31].filled(np.nan).astype(np.float32)
                parts.append(g.reshape(g.shape[0], -1)[:, cells])
            out[var] = np.concatenate(parts)
    dates = np.array([np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}") for t in time])
    np.savez_compressed(os.path.join(FD, f"hyras_cells_{year}.npz"), dates=dates, cells=cells, **out)
    for p in paths:
        os.remove(p)
    print(f"{year}: {len(cells)} cells, {len(dates)} days, rain per cell {np.nanmean(out['pr'].sum(0)):.0f} mm "
          f"(range {np.nanmin(out['pr'].sum(0)):.0f}-{np.nanmax(out['pr'].sum(0)):.0f}), "
          f"mean temperature {np.nanmean(out['tas']):.1f} C", flush=True)
    return cells


def main():
    os.makedirs(RAW, exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[-1])
    cells = None
    for year in range(a, b + 1):
        path = os.path.join(FD, f"hyras_cells_{year}.npz")
        if os.path.exists(path):
            cells = np.load(path)["cells"]
            print(f"{year}: already processed")
            continue
        cells = process_year(year, cells)


if __name__ == "__main__":
    main()
