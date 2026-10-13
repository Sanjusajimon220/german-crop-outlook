"""Sentinel-1 debug test (owner OK 2026-10-13): why did the first field test (phase4_s1_fields.py, 500 fields 2023)
return an empty table after reading 2.4 gigapixels?

20 winter wheat fields of 2024 (Rur, >= 2 ha), WHOLE outlines (no inward shrink), May 2024, VV + VH, sigma0.
Variant A: no orbit filter.  Variant B: orbit filter 'ascending' (lower case, as in the STAC metadata).
Each variant is one small job (about 1 credit). Prints rows, fields with values, dates and the cost.
"""
import json
import os
import sys
import time

import geopandas as gpd
import numpy as np
import openeo
import pandas as pd

OUT = os.path.join("data", "processed", "s1", "debug")


def fields(n=20):
    f = gpd.read_file(os.path.join("data", "processed", "fields_rur.gpkg"))
    f = f[(f["VALIDFROM"] == 2024) & (f["CODE"] == 115) & (f["AREA_HA"] >= 2)].sample(n, random_state=1)
    f = f[["ID", "geometry"]].to_crs(4326)
    f["geometry"] = f.geometry.simplify(0.00002)
    return f


def run(variant, f, con):
    props = {"sat:orbit_state": lambda x: x == "ascending"} if variant == "B" else None
    cube = con.load_collection("SENTINEL1_GRD", temporal_extent=["2024-05-01", "2024-05-31"], bands=["VV", "VH"],
                               spatial_extent=dict(zip(["west", "south", "east", "north"], map(float, f.total_bounds))),
                               properties=props)
    cube = cube.sar_backscatter(coefficient="sigma0-ellipsoid")
    job = cube.aggregate_spatial(geometries=json.loads(f.to_json()), reducer="mean").create_job(
        title=f"s1 debug {variant}", out_format="CSV")
    job.start()
    while job.status() in ("created", "queued", "running"):
        time.sleep(30)
    d = job.describe()
    folder = os.path.join(OUT, variant)
    if job.status() == "finished":
        job.get_results().download_files(folder)
        t = pd.concat([pd.read_csv(os.path.join(folder, x)) for x in os.listdir(folder) if x.endswith(".csv")])
        v = t.dropna(subset=["VV"]) if "VV" in t else t.iloc[:0]
        print(f"variant {variant}: status finished, rows {len(t)}, rows with VV {len(v)}, fields {v['feature_index'].nunique() if len(v) else 0}, "
              f"dates {v['date'].nunique() if len(v) else 0}, credits {d.get('costs')}, columns {list(t.columns)}", flush=True)
        if len(v):
            print(v.assign(VV_db=10 * np.log10(v["VV"]), VH_db=10 * np.log10(v["VH"]))[["VV_db", "VH_db"]].describe().round(2).to_string())
    else:
        print(f"variant {variant}: {job.status()}, credits {d.get('costs')}", flush=True)
        try:
            print(job.logs()[-5:])
        except Exception as e:
            print(e)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    f = fields()
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    for v in sys.argv[1:] or ["A", "B"]:
        run(v, f, con)
