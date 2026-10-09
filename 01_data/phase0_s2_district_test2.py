"""Sentinel-2 district greenness TEST 2 (owner-approved): Thuringia 2022, ONE crop per job.
HR-VPP PPI collection fails server-side (load_stac error, 2026-10-09) -> NDVI from SENTINEL2_L2A:
B04/B08, clouds/shadows/snow masked with SCL (keep 4 vegetation, 5 bare soil), 10-day (dekad) median
composites, masked to the CLMS HRL crop type of the year (1110 wheat, 1120 barley, 1130 maize,
1310 potatoes), mean per district (VG250). Server-side; only the table is downloaded.
Run `python phase0_s2_district_test2.py 1110`."""
import json
import os
import sys

import geopandas as gpd
import openeo

OUT = os.path.join("data", "processed", "s2_district")
code = int(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                    + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GEN", "GF"])
krs = krs[(krs["GF"] == 4) & krs["AGS"].str.startswith("16")].to_crs(4326)[["AGS", "geometry"]]
geoms = json.loads(krs.to_json())
w, s, e, n = krs.total_bounds
bb = dict(west=float(w), south=float(s), east=float(e), north=float(n), crs="EPSG:4326")
con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
s2 = con.load_collection("SENTINEL2_L2A", spatial_extent=bb, temporal_extent=["2022-03-01", "2022-10-31"],
                         bands=["B04", "B08", "SCL"], max_cloud_cover=80)
scl = s2.band("SCL")
bad = ~((scl == 4) | (scl == 5))
ndvi = s2.ndvi(nir="B08", red="B04").mask(bad)
ndvi = ndvi.aggregate_temporal_period(period="dekad", reducer="median")
cty = con.load_collection("CLMS_VLCC_CROP_TYPES_EUROPE_10M_YEARLY_V1", spatial_extent=bb,
                          temporal_extent=["2022-01-01", "2022-12-31"], bands=["cty"]).reduce_dimension(dimension="t", reducer="max")
cty = cty.resample_cube_spatial(ndvi, method="near")
ndvi = ndvi.mask(cty.band("cty") != code)
agg = ndvi.aggregate_spatial(geometries=geoms, reducer="mean")
job = agg.create_job(title=f"vista s2 ndvi thuringia 2022 crop {code}", out_format="CSV")
job.start_and_wait()
job.get_results().download_files(os.path.join(OUT, f"test_16_2022_{code}"))
d = job.describe()
print("status", d.get("status"), "costs", d.get("costs"), "usage", json.dumps(d.get("usage", {}))[:300])
