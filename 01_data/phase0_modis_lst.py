"""District canopy (land surface) temperature 2000-2026: MODIS MOD11A2 (Terra, ~10:30) and MYD11A2
(Aqua, ~13:30) v061, 8-day composites, 1 km (v10, owner-approved; streamed, nothing large stored).

Source: NASA LP DAAC via Microsoft Planetary Computer (open). Tiles h18v03 + h18v04.
  cropland     ESA WorldCover 2021 share of cropland per 1 km pixel >= 50 %
  quality      QC_Day bits 0-1 = 0 (good) or 1 (other, LST produced); LST = DN x 0.02 K
  per composite  district mean daytime LST (deg C) over clear cropland pixels and their number
Months March-October. Saved per sensor and year: data/processed/modis_lst/lst_<MOD|MYD>_<year>.npz
(dates = composite start dates, codes, lst[dates, districts], n[dates, districts]); resumable.
Run `python phase0_modis_lst.py 2000 2026`.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from phase0_modis_districts import cropland_share, read, search, signed

OUT = os.path.join("data", "processed", "modis_lst")
TILES = ("h18v03", "h18v04")
COLL = "modis-11A2-061"
CROP_MIN = 0.5
SENSORS = ("MYD",) if "aqua" in sys.argv else ("MYD", "MOD")   # 'aqua': Aqua only (used by the features)


def tile_setup(tile, codes):
    path = os.path.join(OUT, f"grid_{tile}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["dist"], d["crop"], tuple(d["win"])
    item = search(COLL, "2019-06-01T00:00:00Z/2019-06-30T00:00:00Z", tile)[0]
    with rasterio.open(signed(item["assets"]["LST_Day_1km"]["href"])) as src:
        crs, transform, shape = src.crs, src.transform, src.shape
    krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                        + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GF"])
    krs = krs[(krs["GF"] == 4) & krs["AGS"].isin(codes)].to_crs(crs)
    pos = {c: i for i, c in enumerate(codes)}
    dist = rasterize(((g, pos[a]) for g, a in zip(krs.geometry, krs["AGS"])), out_shape=shape,
                     transform=transform, fill=-1, dtype="int32")
    rows, cols = np.where(dist >= 0)
    win = (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)
    dist = dist[win[0]:win[1], win[2]:win[3]]
    crop = cropland_share(crs, transform, win) >= CROP_MIN
    np.savez(path, dist=dist, crop=crop, win=np.array(win))
    print(f"  {tile}: {np.sum(dist >= 0)} district pixels, {np.sum((dist >= 0) & crop)} cropland", flush=True)
    return dist, crop, win


def main():
    os.makedirs(OUT, exist_ok=True)
    codes = list(np.load(os.path.join("data", "processed", "weather_daily.npz"))["codes"])
    grids = {t: tile_setup(t, codes) for t in TILES}
    yrs = [a for a in sys.argv[1:] if a.isdigit()]
    for year in range(int(yrs[0]), int(yrs[-1]) + 1):
        items = None
        for sensor in SENSORS:
            path = os.path.join(OUT, f"lst_{sensor}_{year}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            if items is None:
                items = {t: search(COLL, f"{year}-03-01T00:00:00Z/{year}-10-31T23:59:59Z", t) for t in TILES}
            its = {t: [f for f in v if f["id"].startswith(sensor + "11A2")] for t, v in items.items()}
            dates = sorted({f["properties"]["start_datetime"][:10] for v in its.values() for f in v})
            if not dates:
                print(f"{sensor} {year}: no data", flush=True)
                continue
            sums = np.zeros((len(dates), len(codes)))
            nn = np.zeros((len(dates), len(codes)), np.int32)
            for t in TILES:
                dist, crop, win = grids[t]
                w = Window(win[2], win[0], win[3] - win[2], win[1] - win[0])

                def get(f):
                    try:
                        return (f, read(f["assets"]["LST_Day_1km"]["href"], COLL, w)[0],
                                read(f["assets"]["QC_Day"]["href"], COLL, w)[0])
                    except RuntimeError as e:          # unreadable file on the server: skip, note
                        print(f"  skipped {f['id']}: {e}", flush=True)
                        return None
                with ThreadPoolExecutor(8) as pool:          # 8 composites read at once
                    results = list(pool.map(get, its[t]))
                for f, lst, qc in [r for r in results if r is not None]:
                    k = dates.index(f["properties"]["start_datetime"][:10])
                    ok = (dist >= 0) & crop & (lst > 0) & ((qc & 3) <= 1)
                    idx = dist[ok]
                    sums[k] += np.bincount(idx, weights=lst[ok] * 0.02 - 273.15, minlength=len(codes))
                    nn[k] += np.bincount(idx, minlength=len(codes))
            lst = np.where(nn > 0, sums / np.maximum(nn, 1), np.nan).astype(np.float32)
            np.savez_compressed(path, dates=np.array(dates, dtype="datetime64[D]"), codes=np.array(codes), lst=lst, n=nn)
            jul = [i for i, d in enumerate(dates) if d[5:7] == "07"]
            print(f"{sensor} {year}: {len(dates)} composites, July mean LST {np.nanmean(lst[jul]) if jul else np.nan:.1f} C, "
                  f"districts with data {np.mean(np.any(nn > 0, 0)) * 100:.0f} % ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
