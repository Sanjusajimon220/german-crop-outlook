"""Landsat 8 + 9 Collection 2 Level-2 per area and season (pre-registered docs/project_log.md 2026-10-16).

Source: Microsoft Planetary Computer STAC, collection landsat-c2-l2 (free; links signed by planetary_computer).
Per scene: bands blue, green, red, nir08, swir16, swir22 (surface reflectance) + qa_pixel, warped to the area's
30 m grid in EPSG:32632 (aligned to the 8 x 8 km box), one GeoTIFF per scene; scene list with sun angles in
scenes.csv. Scenes with > 95 % cloud over the whole scene are skipped (as S2).
Run `python phase10_landsat.py AREA YEAR [AREA YEAR ...]`; resumable (existing files skipped).
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from areas import bounds, period, raw_dir

ASSETS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
RES = 30


def grid(area):
    x0, y0, x1, y1 = bounds(area)
    n = int((x1 - x0) / RES)
    return rasterio.transform.from_origin(x0, y1, RES, RES), n, n


def ok_file(path):
    """An existing, readable file (a run stopped mid-write leaves a broken one)."""
    if not os.path.exists(path):
        return False
    try:
        with rasterio.open(path) as r:
            r.read(r.count, window=((0, 1), (0, 1)))
        return True
    except Exception:
        os.remove(path)
        return False


def one_scene(it, path, transform, w, h):
    """Warp the needed window of each band onto the area grid (only the touched COG tiles are read)."""
    stack = []
    for a in ASSETS:
        for k in range(5):
            try:
                with rasterio.open(it.assets[a].href) as src, WarpedVRT(src, crs="EPSG:32632", transform=transform, width=w,
                                                                       height=h, resampling=Resampling.nearest) as v:
                    stack.append(v.read(1))
                break
            except Exception:
                if k == 4:
                    raise
                time.sleep(10 * (k + 1))
    prof = dict(driver="GTiff", width=w, height=h, count=len(ASSETS), dtype="uint16", crs="EPSG:32632", transform=transform,
                compress="deflate", nodata=0)
    tmp = path + ".part"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(np.stack(stack).astype(np.uint16))
        dst.descriptions = tuple(ASSETS)
    os.replace(tmp, path)                                         # complete files only


def fetch(area, year):
    out = os.path.join(raw_dir(area, year), "landsat")
    os.makedirs(out, exist_ok=True)
    x0, y0, x1, y1 = bounds(area)
    tr = Transformer.from_crs(32632, 4326, always_xy=True)
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    t0, t1 = period(year)
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=[lon0, lat0, lon1, lat1], datetime=f"{t0}/{t1}",
                            query={"platform": {"in": ["landsat-8", "landsat-9"]}, "eo:cloud_cover": {"lt": 95}}).items())
    transform, w, h = grid(area)
    rows, todo = [], []
    for it in sorted(items, key=lambda i: i.datetime):
        name = f"{it.datetime:%Y-%m-%d}_{it.properties['platform']}_{it.id}.tif"
        rows.append(dict(file=name, date=f"{it.datetime:%Y-%m-%d}", time=f"{it.datetime:%H:%M}", platform=it.properties["platform"],
                         scene_cloud=it.properties.get("eo:cloud_cover"), sun_elevation=it.properties.get("view:sun_elevation"),
                         sun_azimuth=it.properties.get("view:sun_azimuth"), off_nadir=it.properties.get("view:off_nadir", 0)))
        path = os.path.join(out, name)
        if not ok_file(path):
            todo.append((it, path))
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                      VSI_CACHE="TRUE", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda a: one_scene(*a, transform, w, h), todo))
    pd.DataFrame(rows).to_csv(os.path.join(out, "scenes.csv"), index=False)
    print(f"{time.strftime('%H:%M')} {area} {year} landsat: {len(rows)} scenes "
          f"({sum(r['platform'] == 'landsat-8' for r in rows)} L8, {sum(r['platform'] == 'landsat-9' for r in rows)} L9)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    for i in range(0, len(a), 2):
        fetch(a[i], int(a[i + 1]))
