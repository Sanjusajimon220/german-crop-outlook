"""District satellite greenness 2000-2026: MODIS Terra NDVI (MOD13Q1 v061, 250 m, 16-day composites).

Source: NASA LP DAAC MODIS (public domain) via Microsoft Planetary Computer (open, no account; cloud-
optimised GeoTIFFs, only the Germany part of each file is read). Terra only (one sensor for all years).
Germany = MODIS tiles h18v03 + h18v04 (sinusoidal grid).
  cropland mask  ESA WorldCover 2021 (10 m, CC BY 4.0; read at its 80 m overview): share of cropland
                 (class 40) per 250 m pixel; pixels with >= 60 % cropland are used
  districts      VG250 districts rasterised onto the tile grid
  per composite  district mean NDVI over cropland pixels with good / marginal reliability (0/1), and
                 the number of such pixels (cloud-free share)
Months February-October. Saved per year: data/processed/modis/modis_ndvi_<year>.npz
(dates = composite start dates, codes, ndvi[dates, districts], n[dates, districts]); resumable.
Run `python phase0_modis_districts.py 2000 2026`.
"""
import os
import sys
import time

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.windows import Window

API = "https://planetarycomputer.microsoft.com/api"
OUT = os.path.join("data", "processed", "modis")
TILES = ("h18v03", "h18v04")
_TOK = {}


def signed(href):
    """Signed read URL for a Planetary Computer asset (the per-collection token was refused)."""
    for attempt in range(5):
        try:
            return requests.get(f"{API}/sas/v1/sign", params={"href": href}, timeout=60).json()["href"]
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("signing failed")


def token(collection):
    t = _TOK.get(collection)
    if t is None or time.time() > t[1]:
        tok = requests.get(f"{API}/sas/v1/token/{collection}", timeout=60).json()["token"]
        _TOK[collection] = (tok, time.time() + 1800)
    return _TOK[collection][0]


def search(collection, dt, tile):
    h, v = int(tile[1:3]), int(tile[4:6])
    feats, body = [], {"collections": [collection], "datetime": dt, "limit": 200,
                       "query": {"modis:horizontal-tile": {"eq": h}, "modis:vertical-tile": {"eq": v}}}
    url = f"{API}/stac/v1/search"
    while True:
        r = requests.post(url, json=body, timeout=120).json()
        feats += r.get("features", [])
        nxt = [l for l in r.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            return feats
        body = nxt[0].get("body", body)


def read(href, collection, window=None):
    for attempt in range(5):
        try:
            with rasterio.open(signed(href)) as src:
                return src.read(1, window=window), src
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"cannot read {href}")


CROP_MIN = 0.6      # a MODIS pixel counts as cropland if >= 60 % of it is cropland (WorldCover class 40)


def cropland_share(crs, transform, win):
    """Share of ESA WorldCover 2021 cropland (10 m, read at its 80 m overview) in each 250 m pixel."""
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import Affine
    h, w = win[1] - win[0], win[3] - win[2]
    dst_t = transform * Affine.translation(win[2], win[0])
    share = np.full((h, w), np.nan, np.float32)
    feats = requests.post(f"{API}/stac/v1/search", json={"collections": ["esa-worldcover"], "bbox": [5.5, 47.0, 15.5, 55.2],
                          "query": {"esa_worldcover:product_version": {"eq": "2.0.0"}}, "limit": 50}, timeout=120).json()["features"]
    for f in feats:
        with rasterio.open(signed(f["assets"]["map"]["href"]), overview_level=2) as src:
            a = (src.read(1) == 40).astype(np.float32)
            part = np.full((h, w), np.nan, np.float32)
            reproject(a, part, src_transform=src.transform, src_crs=src.crs, dst_transform=dst_t, dst_crs=crs,
                      resampling=Resampling.average, src_nodata=None, dst_nodata=np.nan)
        share = np.fmax(share, part)
    print(f"  WorldCover: {len(feats)} tiles merged", flush=True)
    return np.nan_to_num(share)


def tile_setup(tile, codes):
    """District index and cropland mask on the 250 m grid of one tile, cropped to Germany."""
    path = os.path.join(OUT, f"grid_{tile}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["dist"], d["crop"], tuple(d["win"])
    item = search("modis-13Q1-061", "2019-06-01T00:00:00Z/2019-06-30T00:00:00Z", tile)[0]
    href = item["assets"]["250m_16_days_NDVI"]["href"]
    with rasterio.open(signed(href)) as src:
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
    for year in range(int(sys.argv[1]), int(sys.argv[-1]) + 1):
        path = os.path.join(OUT, f"modis_ndvi_{year}.npz")
        if os.path.exists(path):
            print(f"{year}: done before")
            continue
        t0 = time.time()
        items = {t: search("modis-13Q1-061", f"{year}-02-01T00:00:00Z/{year}-10-31T23:59:59Z", t) for t in TILES}
        items = {t: [f for f in v if f["id"].startswith("MOD13Q1")] for t, v in items.items()}
        dates = sorted({f["properties"]["start_datetime"][:10] for v in items.values() for f in v})
        nd = np.full((len(dates), len(codes)), np.nan, np.float32)
        nn = np.zeros((len(dates), len(codes)), np.int32)
        sums = np.zeros((len(dates), len(codes)))
        for t in TILES:
            dist, crop, win = grids[t]
            w = Window(win[2], win[0], win[3] - win[2], win[1] - win[0])
            for f in items[t]:
                k = dates.index(f["properties"]["start_datetime"][:10])
                ndvi, _ = read(f["assets"]["250m_16_days_NDVI"]["href"], "modis-13Q1-061", w)
                rel, _ = read(f["assets"]["250m_16_days_pixel_reliability"]["href"], "modis-13Q1-061", w)
                ok = (dist >= 0) & crop & (rel >= 0) & (rel <= 1) & (ndvi > -2000)
                idx = dist[ok]
                sums[k] += np.bincount(idx, weights=ndvi[ok] * 1e-4, minlength=len(codes))
                nn[k] += np.bincount(idx, minlength=len(codes))
        nd = np.where(nn > 0, sums / np.maximum(nn, 1), np.nan).astype(np.float32)
        np.savez_compressed(path, dates=np.array(dates, dtype="datetime64[D]"), codes=np.array(codes), ndvi=nd, n=nn)
        print(f"{year}: {len(dates)} composites, median NDVI {np.nanmedian(nd):.3f}, districts with data "
              f"{np.mean(np.any(nn > 0, 0)) * 100:.0f} % ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
