"""Sentinel-2 crop-specific district greenness by SAMPLING (v11 groundwork; owner-approved test).

Full-coverage test failed (cluster errors; 39 credits, 37 gigapixels for one state / crop / year).
Design: CLMS HRL crop types (cty) resampled to 100 m (mode) -> 'interior' cells (cell and its 8 neighbours
the same crop) -> per district and crop up to N random cells -> 40 m squares at the cell centres ->
one MultiPolygon per district x crop -> openEO: Sentinel-2 L2A NDVI (B04/B08), SCL keep 4/5, mean over each
MultiPolygon for every acquisition date (no server-side temporal compositing); 10-day composites made
locally (median of valid dates). Only the table is downloaded.
Crops: 1110 wheat, 1120 barley, 1130 maize, 1310 potatoes.
Run `python phase0_s2_district_sample.py <state code> <year> [N] [cheap]`; 'cheap' = April-September and
scenes with <= 60 % cloud cover (results in sample_<state>_<year>_n<N>_cheap).
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import openeo
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import minimum_filter, maximum_filter
from shapely.geometry import MultiPolygon, box

OUT = os.path.join("data", "processed", "s2_district")
CROPS = {1110: "wheat", 1120: "barley", 1130: "maize", 1310: "potato"}
# 'dlr' option: DLR CropTypes V02 (local file data/raw/croptypes_dlr, CC BY 4.0; winter wheat / winter barley only)
CROPS_DLR = {11: "wheat", 12: "barley", 30: "maize", 50: "potato"}


def dlr_cty100(krs, year, tif):
    """DLR CropTypes 10 m -> 100 m (mode) for the state's bounding box (EPSG:32632)."""
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    src_f = os.path.join("data", "raw", "croptypes_dlr", f"CROPTYPES_DE_P1Y_{year}_V02.tif")
    with rasterio.open(src_f) as src:
        w, s, e, n = krs.to_crs(src.crs).total_bounds
        win = from_bounds(w, s, e, n, src.transform).round_offsets().round_lengths()
        h, wd = int(win.height // 10), int(win.width // 10)
        a = src.read(1, window=win, out_shape=(h, wd), resampling=Resampling.mode)
        tr = src.window_transform(win) * rasterio.Affine.scale(win.width / wd, win.height / h)
        prof = dict(driver="GTiff", height=h, width=wd, count=1, dtype=a.dtype, crs=src.crs, transform=tr, compress="deflate")
    with rasterio.open(tif, "w", **prof) as dst:
        dst.write(a, 1)


def retry(fn, tries=8):
    """Server rate limit (429 Too Many Requests): wait and try again."""
    import time
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if "429" not in str(e) and "Too Many" not in str(e) or k == tries - 1:
                raise
            time.sleep(30 * (k + 1))


def main():
    state, year = sys.argv[1], int(sys.argv[2])
    n_cells = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    cheap = "cheap" in sys.argv
    dlr = "dlr" in sys.argv
    crops = CROPS_DLR if dlr else CROPS
    if dlr and year >= 2024:      # 2026-10-10: replaced by the per-cell design (phase0_s2_cells.py, owner OK)
        print("skipped: 2024+ DLR sampling replaced by phase0_s2_cells.py", flush=True)
        return
    tag = f"{state}_{year}" + (f"_n{n_cells}_cheap" if cheap else "") + ("_dlr" if dlr else "")
    months, cloud = (("04-01", "09-30"), 60) if cheap else (("03-01", "10-31"), 80)
    os.makedirs(OUT, exist_ok=True)
    krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                        + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GF"])
    krs = krs[(krs["GF"] == 4) & krs["AGS"].str.startswith(state)]
    con = retry(lambda: openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc())
    k4326 = krs.to_crs(4326)
    w, s, e, n = k4326.total_bounds
    bb = dict(west=float(w), south=float(s), east=float(e), north=float(n), crs="EPSG:4326")
    tif = os.path.join(OUT, f"cty100_{state}_{year}" + ("_dlr" if dlr else "") + ".tif")
    if dlr and not os.path.exists(tif):
        dlr_cty100(krs, year, tif)
    if not os.path.exists(tif):
        c = con.load_collection("CLMS_VLCC_CROP_TYPES_EUROPE_10M_YEARLY_V1", spatial_extent=bb,
                                temporal_extent=[f"{year}-01-01", f"{year}-12-31"], bands=["cty"]).reduce_dimension(dimension="t", reducer="max")
        c = c.resample_spatial(resolution=100, projection=3035, method="mode")
        job = retry(lambda: c.create_job(title=f"vista cty100 {state} {year}", out_format="GTiff"))
        job.start_and_wait()                                   # batch: sync mode is limited to 20000 px
        tmp = os.path.join(OUT, f"cty100_{state}_{year}_job")
        job.get_results().download_files(tmp)
        got = [f for f in os.listdir(tmp) if f.endswith(".tif")]
        os.replace(os.path.join(tmp, got[0]), tif)
    with rasterio.open(tif) as src:
        a, tr, crs = src.read(1), src.transform, src.crs
    kk = krs.to_crs(crs)
    dist = rasterize(((g, i) for i, g in enumerate(kk.geometry)), out_shape=a.shape, transform=tr, fill=-1, dtype="int32")
    interior = (minimum_filter(a, 3) == a) & (maximum_filter(a, 3) == a)
    rng = np.random.default_rng(year * 100 + int(state))
    rows = []
    for i, ags in enumerate(kk["AGS"]):
        for code, name in crops.items():
            r, c = np.where((dist == i) & interior & (a == code))
            if len(r) == 0:
                continue
            pick = rng.choice(len(r), size=min(n_cells, len(r)), replace=False)
            sq = []
            for j in pick:
                x, y = rasterio.transform.xy(tr, r[j], c[j])
                sq.append(box(x - 20, y - 20, x + 20, y + 20))
            rows.append(dict(AGS=ags, crop=name, cells=len(pick), available=len(r), geometry=MultiPolygon(sq)))
    g = gpd.GeoDataFrame(rows, crs=crs).to_crs(4326)
    g.drop(columns="geometry").to_csv(os.path.join(OUT, f"sample_{tag}.csv"), index=False)
    print(f"{len(g)} district x crop samples, cells: " + ", ".join(f"{k} {v}" for k, v in g.groupby("crop")["cells"].sum().items()), flush=True)
    s2 = con.load_collection("SENTINEL2_L2A", spatial_extent=bb, temporal_extent=[f"{year}-{months[0]}", f"{year}-{months[1]}"],
                             bands=["B04", "B08", "SCL"], max_cloud_cover=cloud)
    scl = s2.band("SCL")
    ndvi = s2.ndvi(nir="B08", red="B04").mask(~((scl == 4) | (scl == 5)))
    agg = ndvi.aggregate_spatial(geometries=json.loads(g[["AGS", "crop", "geometry"]].to_json()), reducer="mean")
    job = retry(lambda: agg.create_job(title=f"vista s2 sample {tag}", out_format="CSV"))
    job.start_and_wait()
    job.get_results().download_files(os.path.join(OUT, f"sample_{tag}"))
    d = job.describe()
    print("status", d.get("status"), "credits", d.get("costs"),
          "input megapixel", (d.get("usage") or {}).get("input_pixel", {}).get("value"), flush=True)


if __name__ == "__main__":
    main()
