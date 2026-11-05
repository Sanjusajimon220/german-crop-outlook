"""Landsat 8 / 9 Collection 2 Level-2 surface temperature per area and season, 1 April - 30 September
(thermal water method feasibility, docs/project_log.md 2026-11-05).

Source: Microsoft Planetary Computer STAC, collection landsat-c2-l2 (free). Per scene, on the area's 30 m grid (same grid
and warping as phase10_landsat): lwir11 (ST_B10, K = DN x 0.00341802 + 149), qa (ST uncertainty, K = DN x 0.01) and
qa_pixel (cloud mask; also needed for September, which the reflectance download does not cover).
Run from D:\\vista: python phase31_landsat_thermal.py AREA YEAR [AREA YEAR ...] -> data/raw/field_v2/AREA_YEAR/landsat_st/
Resumable (existing complete files skipped).
"""
import os
import sys
import time

import numpy as np

ASSETS = ["lwir11", "qa", "qa_pixel"]
ST_SCALE, ST_OFFSET = 0.00341802, 149.0


def window(year):
    return f"{year}-04-01", f"{year}-09-30"


def kelvin(dn):
    """Surface temperature (K) from ST_B10 digital numbers; 0 = no data."""
    dn = np.asarray(dn)
    return np.where(dn == 0, np.nan, dn.astype(float) * ST_SCALE + ST_OFFSET)


def fetch(area, year):
    import pandas as pd
    import planetary_computer
    import pystac_client
    import phase10_landsat as L
    from concurrent.futures import ThreadPoolExecutor
    from pyproj import Transformer
    import rasterio
    from areas import bounds, raw_dir
    out = os.path.join(raw_dir(area, year), "landsat_st")
    os.makedirs(out, exist_ok=True)
    x0, y0, x1, y1 = bounds(area)
    tr = Transformer.from_crs(32632, 4326, always_xy=True)
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    t0, t1 = window(year)
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=[lon0, lat0, lon1, lat1], datetime=f"{t0}/{t1}",
                            query={"platform": {"in": ["landsat-8", "landsat-9"]}, "eo:cloud_cover": {"lt": 95}}).items())
    transform, w, h = L.grid(area)
    L.ASSETS = ASSETS                                            # same warping code, thermal assets
    rows, todo = [], []
    for it in sorted(items, key=lambda i: i.datetime):
        name = f"{it.datetime:%Y-%m-%d}_{it.properties['platform']}_{it.id}.tif"
        rows.append(dict(file=name, date=f"{it.datetime:%Y-%m-%d}", time=f"{it.datetime:%H:%M}",
                         platform=it.properties["platform"], scene_cloud=it.properties.get("eo:cloud_cover")))
        path = os.path.join(out, name)
        if not L.ok_file(path):
            todo.append((it, path))
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                      VSI_CACHE="TRUE", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with ThreadPoolExecutor(4) as ex:
            failed = [f for f in ex.map(lambda a: L.safe_scene(*a, transform, w, h), todo) if f]
    if failed:
        print(f"{area} {year}: skipped {len(failed)} unreadable scene(s): {failed}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "scenes.csv"), index=False)
    print(f"{time.strftime('%H:%M')} {area} {year} thermal: {len(rows)} scenes, {len(todo)} new", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    for i in range(0, len(a), 2):
        fetch(a[i], int(a[i + 1]))
