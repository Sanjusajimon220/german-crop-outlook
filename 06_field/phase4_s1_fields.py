"""Phase 4, step 1b: Sentinel-1 radar time series per field (Copernicus Data Space, openEO).

Radar sees through clouds: backscatter in two polarisations, VV and VH, reacts to plant water,
biomass and structure (VH/VV rises with canopy volume; drops at heading / ripening).
Processing on the server: SENTINEL1_GRD -> sar_backscatter (sigma0, terrain-flattened ellipsoid
normalisation; the local incidence angle is not offered for this collection on CDSE) -> mean per field (same 10 m
inward-shrunk outlines as the Sentinel-2 extraction) -> linear values; dB computed locally.
One orbit direction per job (ascending) so the viewing geometry stays comparable.
Run `python phase4_s1_fields.py test` (500 wheat fields, season 2022/23; prints cost) or
`python phase4_s1_fields.py <crop code> <harvest year> [max fields]`.
Writes data/processed/s1/s1_<code>_<year>[_<n>fields].csv.
"""
import os
import sys

import numpy as np
import openeo
import pandas as pd

from phase4_s2_fields import fields_geojson

OUT = os.path.join("data", "processed", "s1")


def run(code, year, limit, orbit="ASCENDING"):
    os.makedirs(OUT, exist_ok=True)
    gj, _ = fields_geojson(code, year, limit)
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    title = f"field S1 {code} {year} {limit}"
    old = [j for j in con.list_jobs() if j.get("title") == title and j.get("status") in ("finished", "running", "queued")]
    if old:
        job = con.job(old[0]["id"])
    else:
        cube = con.load_collection("SENTINEL1_GRD", temporal_extent=[f"{year - 1}-10-01", f"{year}-08-31"],
                                   bands=["VV", "VH"],
                                   properties={"sat:orbit_state": lambda x: x == orbit})
        cube = cube.sar_backscatter(coefficient="sigma0-ellipsoid")   # local incidence angle not offered on CDSE
        job = cube.aggregate_spatial(geometries=gj, reducer="mean").create_job(title=title, out_format="CSV")
        job.start()
    import time
    while job.status() in ("created", "queued", "running"):
        time.sleep(60)
    info = job.describe()
    if job.status() != "finished":
        raise RuntimeError(f"job {job.status()}")
    folder = os.path.join(OUT, f"job_{code}_{year}_{limit}")
    job.get_results().download_files(folder)
    df = pd.concat([pd.read_csv(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".csv")])
    ids = [f["properties"]["ID"] for f in gj["features"]]
    df["field_id"] = df["feature_index"].map(dict(enumerate(ids)))
    for b in ("VV", "VH"):
        df[f"{b}_db"] = 10 * np.log10(df[b].where(df[b] > 0))
    df["VH_VV_db"] = df["VH_db"] - df["VV_db"]
    name = f"s1_{code}_{year}" + (f"_{limit}fields" if limit < 100000 else "") + ".csv"
    df.to_csv(os.path.join(OUT, name), index=False)
    v = df.dropna(subset=["VV"])
    print(f"{len(df)} rows, {df['field_id'].nunique()} fields, {v['date'].nunique()} dates, "
          f"median obs per field {v.groupby('field_id').size().median():.0f}")
    print(f"usage: {info.get('usage')}; costs: {info.get('costs')}")
    m = v.assign(month=v["date"].str[:7]).groupby("month")[["VV_db", "VH_db", "VH_VV_db"]].median().round(2)
    print(m.to_string())


if __name__ == "__main__":
    if sys.argv[1] == "test":
        run(115, 2023, 500)
    else:
        run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 100000)
