"""Phase 0: soil water properties per district from the DWD agrometeorological soil maps.

Input  data/raw/soil/AG_SOILINFO_THETAFC.nc, AG_SOILINFO_THETAWP.nc
       1 km grids for 20 soil layers of 10 cm (0-2 m) used by the DWD's AMBAV model:
       volumetric water content at field capacity (FC: what the soil holds after draining)
       and at the wilting point (WP: below this roots cannot extract water)
Output data/processed/soil_districts.csv: per district and layer the mean FC and WP, and
       the plant-available water (FC - WP) summed over 0-1 m and 0-2 m, in mm.

Run `python phase0_soil.py`.
"""
import os

import netCDF4
import numpy as np
import pandas as pd

from weather_districts import district_grid

RAW = os.path.join("data", "raw", "soil")
OUT = os.path.join("data", "processed", "soil_districts.csv")
CACHE = os.path.join("data", "processed", "district_grid_soil.npz")
LAYER_MM = 100.0


def main():
    with netCDF4.Dataset(os.path.join(RAW, "AG_SOILINFO_THETAFC.nc")) as d:
        fc = d["smv"][:].filled(np.nan)
        lat, lon = d["lat"][:], d["lon"][:]
    with netCDF4.Dataset(os.path.join(RAW, "AG_SOILINFO_THETAWP.nc")) as d:
        wp = d["smv"][:].filled(np.nan)
    index, codes, names = district_grid(lat, lon, CACHE)
    idx = index.ravel()
    rows = []
    for k, (code, name) in enumerate(zip(codes, names)):
        cells = idx == k
        f = np.nanmean(fc.reshape(fc.shape[0], -1)[:, cells], 1)
        w = np.nanmean(wp.reshape(wp.shape[0], -1)[:, cells], 1)
        row = {"district": code, "name": name, "cells": int(cells.sum()),
               "paw_1m_mm": float(((f - w) * LAYER_MM)[:10].sum()),
               "paw_2m_mm": float(((f - w) * LAYER_MM).sum())}
        row.update({f"fc_{i + 1}": round(float(v), 4) for i, v in enumerate(f)})
        row.update({f"wp_{i + 1}": round(float(v), 4) for i, v in enumerate(w)})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    paw = table["paw_1m_mm"]
    print(f"{len(table)} districts saved to {OUT}")
    print(f"plant-available water in the top 1 m: median {paw.median():.0f} mm, "
          f"10-90 % range {paw.quantile(0.1):.0f}-{paw.quantile(0.9):.0f} mm")
    for label, sel in (("lowest", table.nsmallest(3, "paw_1m_mm")),
                       ("highest", table.nlargest(3, "paw_1m_mm"))):
        print(f"  {label}: " + ", ".join(f"{r['name']} {r['paw_1m_mm']:.0f} mm" for _, r in sel.iterrows()))


if __name__ == "__main__":
    main()
