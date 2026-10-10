"""v12 A: large crop-specific Sentinel-2 samples on each year's DLR CropTypes map (pre-registered
docs/project_log.md 2026-10-10 18:28).

Per district: up to 100 interior 100 m cells each of winter wheat (11), winter barley (12), maize (30) and 50 of
potato (50), chosen on the DLR map OF THAT YEAR; per-cell S2 L2A NDVI (SCL 4/5, Apr-Sep, cloud <= 60 %), mean of
each 40 m square. Jobs per group of whole districts (<= CHUNK cells) with the group's own bounding box.
Output s2_district/big_<state>_<year>.csv (cells: AGS, crop, x, y) and big_<state>_<year>/timeseries.csv
(feature_index = row of the cell table). Run `python phase0_s2_big.py <state> <year>`.
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

from phase0_s2_cells import s2_ndvi
from phase0_s2_district_sample import OUT, dlr_cty100, retry

N = {11: ("wheat", 100), 12: ("barley", 100), 30: ("maize", 100), 50: ("potato", 50)}
CHUNK = 4000


def main():
    state, year = sys.argv[1], int(sys.argv[2])
    krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                        + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GF"])
    krs = krs[(krs["GF"] == 4) & krs["AGS"].str.startswith(state)]
    cf = os.path.join(OUT, f"big_{state}_{year}.csv")
    if os.path.exists(cf):
        d = pd.read_csv(cf, dtype={"AGS": str})
    else:
        tif = os.path.join(OUT, f"cty100_{state}_{year}_dlr.tif")
        if not os.path.exists(tif):
            dlr_cty100(krs, year, tif)
        with rasterio.open(tif) as src:
            a, tr, crs = src.read(1), src.transform, src.crs
        kk = krs.to_crs(crs)
        dist = rasterize(((g, i) for i, g in enumerate(kk.geometry)), out_shape=a.shape, transform=tr, fill=-1, dtype="int32")
        interior = (minimum_filter(a, 3) == a) & (maximum_filter(a, 3) == a)
        rng = np.random.default_rng(year * 1000 + int(state) + 7)
        rows = []
        for i, ags in enumerate(kk["AGS"]):
            for code, (name, n) in N.items():
                r, c = np.where((dist == i) & interior & (a == code))
                for j in (rng.choice(len(r), size=min(n, len(r)), replace=False) if len(r) else []):
                    x, y = rasterio.transform.xy(tr, r[j], c[j])
                    rows.append(dict(AGS=ags, crop=name, x=x, y=y))
        d = pd.DataFrame(rows)
        d.to_csv(cf, index=False)
    print(f"{len(d)} cells: " + ", ".join(f"{k} {v}" for k, v in d["crop"].value_counts().items()), flush=True)
    g = gpd.GeoDataFrame({"cell": np.arange(len(d)), "AGS": d["AGS"]},
                         geometry=[box(x - 20, y - 20, x + 20, y + 20) for x, y in zip(d["x"], d["y"])], crs=32632).to_crs(4326)
    # groups of whole districts, <= CHUNK cells each
    groups, cur, n = [], [], 0
    for ags, k in g.groupby("AGS", sort=True).size().items():
        if cur and n + k > CHUNK:
            groups.append(cur); cur, n = [], 0
        cur.append(ags); n += k
    groups.append(cur)
    con = retry(lambda: openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc())
    outdir = os.path.join(OUT, f"big_{state}_{year}")
    parts = []
    for p, ags_list in enumerate(groups):
        gp = g[g["AGS"].isin(ags_list)]
        pdir = f"{outdir}_part{p}"
        if not os.path.exists(os.path.join(pdir, "timeseries.csv")):
            w, s, e, nn = gp.total_bounds
            nd = s2_ndvi(con, dict(west=float(w), south=float(s), east=float(e), north=float(nn), crs="EPSG:4326"), year)
            agg = nd.aggregate_spatial(geometries=json.loads(gp[["cell", "geometry"]].to_json()), reducer="mean")
            job = retry(lambda: agg.create_job(title=f"vista s2 big {state} {year} {p}", out_format="CSV"))
            job.start_and_wait()
            job.get_results().download_files(pdir)
            dd = job.describe()
            print("status", dd.get("status"), "credits", dd.get("costs"),
                  "input megapixel", (dd.get("usage") or {}).get("input_pixel", {}).get("value"), flush=True)
        parts.append((gp["cell"].values, pdir))
    ts = pd.concat([pd.read_csv(os.path.join(pdir, "timeseries.csv")).assign(
        feature_index=lambda x, c=cells: c[x["feature_index"].values]) for cells, pdir in parts])
    os.makedirs(outdir, exist_ok=True)
    ts.to_csv(os.path.join(outdir, "timeseries.csv"), index=False)


if __name__ == "__main__":
    main()
