"""S1 sample for gap filling (project log 2026-10-13): up to 300 winter wheat fields 2024 (>= 1 ha, whole outlines) from
the 8 x 8 km cell with the most such fields (the CDSE S1 chain fails for spread-out field sets), Oct 2023 - Aug 2024,
VV/VH sigma0, ascending orbit, batch JSON -> data/processed/s1/s1_115_2024_300fields.csv (field_id, date, VV, VH)."""
import json, os
import geopandas as gpd, numpy as np, openeo, pandas as pd

OUT = os.path.join("data", "processed", "s1")
f = gpd.read_file(os.path.join("data", "processed", "fields_rur.gpkg"))
f = f[(f["VALIDFROM"] == 2024) & (f["CODE"] == 115) & (f["AREA_HA"] >= 1)].copy()
c = f.geometry.centroid
f["cell"] = (c.x // 8000).astype(int).astype(str) + "_" + (c.y // 8000).astype(int).astype(str)
best = f["cell"].value_counts().index[0]
s = f[f["cell"] == best].sort_values("AREA_HA", ascending=False).head(300)
print("cell", best, "fields", len(s), "area ha", round(s["AREA_HA"].sum()))
g = s[["ID", "geometry"]].to_crs(4326)
w, so, e, n = g.total_bounds
con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
cube = con.load_collection("SENTINEL1_GRD", spatial_extent=dict(west=w, south=so, east=e, north=n),
                           temporal_extent=["2023-10-01", "2024-08-31"], bands=["VV", "VH"],
                           properties={"sat:orbit_state": lambda x: x == "ascending"}).sar_backscatter(coefficient="sigma0-ellipsoid")
job = cube.aggregate_spatial(geometries=json.loads(g.to_json()), reducer="mean").create_job(title="s1 sample 300 2024", out_format="JSON")
job.start_and_wait()
folder = os.path.join(OUT, "job_sample_2024")
job.get_results().download_files(folder)
r = json.load(open(os.path.join(folder, "timeseries.json")))
ids = list(g["ID"])
rows = [dict(field_id=ids[i], date=d, VV=v[0], VH=v[1]) for d, vals in r.items() for i, v in enumerate(vals) if v and v[0] is not None]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "s1_115_2024_300fields.csv"), index=False)
print("rows", len(df), "fields", df["field_id"].nunique() if len(df) else 0, "dates", df["date"].nunique() if len(df) else 0,
      "credits", job.describe().get("costs"))
