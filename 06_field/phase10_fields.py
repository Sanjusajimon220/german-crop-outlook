"""Step 1 (fields + exact boundary on the 10 m grid, NRW register, version A only) for every area and season
(project log 2026-10-16). Same method as phase7_step1_fields.version_a. Run `python phase10_fields.py [AREA YEAR ...]`
(default: all areas and seasons)."""
import json
import os
import sys

import rasterio

import phase7_step1_fields as s1
from areas import AREAS, SEASONS, bounds, out_dir


def run(area, year):
    x0, y0, x1, y1 = bounds(area)
    s1.X0, s1.Y0, s1.X1, s1.Y1, s1.YEAR = x0, y0, x1, y1, year
    s1.NR, s1.NC = (y1 - y0) // s1.RES, (x1 - x0) // s1.RES
    s1.TRANSFORM = rasterio.transform.from_origin(x0, y1, s1.RES, s1.RES)
    out = out_dir(area, year)
    os.makedirs(out, exist_ok=True)
    px, fl = s1.version_a()
    px.to_csv(os.path.join(out, "pixels_nrw.csv.gz"), index=False)
    fl.to_csv(os.path.join(out, "fields_nrw.csv"), index=False)
    json.dump(dict(crs="EPSG:25832", origin_x=x0, origin_y=y1, res=s1.RES, rows=s1.NR, cols=s1.NC, season=year, area=area),
              open(os.path.join(out, "grid_info.json"), "w"), indent=1)
    print(f"{area} {year}: {len(fl)} fields, {fl['area_ha'].sum():.0f} ha; pixels {len(px)}; area check median |diff| "
          f"{fl['area_diff_pct'].abs().median():.2f} %; crops {fl.groupby('crop_group')['area_ha'].sum().nlargest(4).round(0).to_dict()}",
          flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    pairs = [(a[i], int(a[i + 1])) for i in range(0, len(a), 2)] or [(ar, y) for ar in AREAS for y in SEASONS[ar] if (ar, y) != ("pilot", 2024)]
    for ar, y in pairs:
        run(ar, y)
