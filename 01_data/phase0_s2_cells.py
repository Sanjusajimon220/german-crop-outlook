"""v11.1 item 2: Sentinel-2 NDVI per CELL at fixed arable locations, for our own crop classification 2025/2026.

Cells (fixed for all years, file s2_district/cells_<state>.csv): from the DLR CropTypes 2024 map at 100 m (mode),
interior cells (3 x 3 same class), per district up to 25 cells each of winter wheat (11), winter barley (12),
maize (30), potato (50) and up to 30 cells of other arable classes (13 14 21 22 23 40 41 42 60 71 72 84).
Labels: DLR class of the cell in 2024 (and 2023 where that map is used for the transfer check).
Per year: S2 L2A NDVI (SCL 4/5), Apr-Sep, cloud <= 60 %, mean of each 40 m square separately (one feature per
cell) -> s2_district/cells_<state>_<year>/timeseries.csv. Same S2 settings as the 'cheap' district samples, so
the district crop curve = mean over the cells of a crop reproduces the district sampling design.
Run `python phase0_s2_cells.py <state> <year>`.
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import maximum_filter, minimum_filter
from shapely.geometry import box

from phase0_s2_district_sample import OUT, dlr_cty100, retry

MAIN = {11: 25, 12: 25, 30: 25, 50: 25}
OTHER = (13, 14, 21, 22, 23, 40, 41, 42, 60, 71, 72, 84)
N_OTHER = 30


def label(path, xy):
    with rasterio.open(path) as src:
        return np.array([v[0] for v in src.sample(xy)])


def cells(state, krs):
    f = os.path.join(OUT, f"cells_{state}.csv")
    if os.path.exists(f):
        return pd.read_csv(f, dtype={"AGS": str})
    tif = os.path.join(OUT, f"cty100_{state}_2024_dlr.tif")
    if not os.path.exists(tif):
        dlr_cty100(krs, 2024, tif)
    with rasterio.open(tif) as src:
        a, tr, crs = src.read(1), src.transform, src.crs
    kk = krs.to_crs(crs)
    dist = rasterize(((g, i) for i, g in enumerate(kk.geometry)), out_shape=a.shape, transform=tr, fill=-1, dtype="int32")
    interior = (minimum_filter(a, 3) == a) & (maximum_filter(a, 3) == a)
    rng = np.random.default_rng(2024 * 100 + int(state))
    rows = []
    for i, ags in enumerate(kk["AGS"]):
        groups = [((a == k), n) for k, n in MAIN.items()] + [(np.isin(a, OTHER), N_OTHER)]
        for mask, n in groups:
            r, c = np.where((dist == i) & interior & mask)
            for j in rng.choice(len(r), size=min(n, len(r)), replace=False) if len(r) else []:
                x, y = rasterio.transform.xy(tr, r[j], c[j])
                rows.append(dict(AGS=ags, x=x, y=y, label2024=int(a[r[j], c[j]])))
    d = pd.DataFrame(rows)
    d.to_csv(f, index=False)
    return d


def main():
    state, year = sys.argv[1], int(sys.argv[2])
    krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                        + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GF"])
    krs = krs[(krs["GF"] == 4) & krs["AGS"].str.startswith(state)]
    d = cells(state, krs)
    dlr = os.path.join("data", "raw", "croptypes_dlr", f"CROPTYPES_DE_P1Y_{year}_V02.tif")
    if os.path.exists(dlr) and f"label{year}" not in d.columns:      # 10 m label at the cell centre
        d[f"label{year}"] = label(dlr, list(zip(d["x"], d["y"])))
        d.to_csv(os.path.join(OUT, f"cells_{state}.csv"), index=False)
    print(f"{len(d)} cells; 2024 labels: " + ", ".join(f"{k} {v}" for k, v in d["label2024"].value_counts().head(8).items()), flush=True)
    g = gpd.GeoDataFrame({"cell": np.arange(len(d))}, geometry=[box(x - 20, y - 20, x + 20, y + 20) for x, y in zip(d["x"], d["y"])],
                         crs=32632).to_crs(4326)
    w, s, e, n = krs.to_crs(4326).total_bounds
    bb = dict(west=float(w), south=float(s), east=float(e), north=float(n), crs="EPSG:4326")
    con = retry(lambda: openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc())
    s2 = con.load_collection("SENTINEL2_L2A", spatial_extent=bb, temporal_extent=[f"{year}-04-01", f"{year}-09-30"],
                             bands=["B04", "B08", "SCL"], max_cloud_cover=60)
    scl = s2.band("SCL")
    ndvi = s2.ndvi(nir="B08", red="B04").mask(~((scl == 4) | (scl == 5)))
    agg = ndvi.aggregate_spatial(geometries=json.loads(g.to_json()), reducer="mean")
    job = retry(lambda: agg.create_job(title=f"vista s2 cells {state} {year}", out_format="CSV"))
    job.start_and_wait()
    job.get_results().download_files(os.path.join(OUT, f"cells_{state}_{year}"))
    dd = job.describe()
    print("status", dd.get("status"), "credits", dd.get("costs"),
          "input megapixel", (dd.get("usage") or {}).get("input_pixel", {}).get("value"), flush=True)


if __name__ == "__main__":
    main()
