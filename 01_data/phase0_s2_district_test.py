"""Sentinel-2 district greenness, TEST JOB (owner-approved 2026-10-09): Thuringia (16), year 2022.

openEO on Copernicus Data Space (server-side; only the result table is downloaded):
  PPI      COPERNICUS_PLANT_PHENOLOGY_INDEX (HR-VPP seasonal trajectories, 10 m, 10-daily, gap-filled)
  classes  CLMS_VLCC_CROP_TYPES_EUROPE_10M_YEARLY_V1 (HRL Croplands crop types, 10 m), codes 1..20
For every class code a band 'ppi_c<code>' = PPI where crop type == code, plus 'n_c<code>' = pixel count;
aggregated (mean / sum) over each district polygon (VG250). Codes are identified afterwards from area
and seasonal shape (no documentation of the codes was reachable).
Writes data/processed/s2/test_thuringia_2022.json + .csv; prints the credit cost if reported.
"""
import json
import os

import geopandas as gpd
import openeo

OUT = os.path.join("data", "processed", "s2")
CODES = range(1, 21)


def main():
    os.makedirs(OUT, exist_ok=True)
    krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                        + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GEN", "GF"])
    krs = krs[(krs["GF"] == 4) & krs["AGS"].str.startswith("16")].to_crs(4326)[["AGS", "GEN", "geometry"]]
    geoms = json.loads(krs.to_json())
    w, s, e, n = krs.total_bounds
    bbox = dict(west=float(w), south=float(s), east=float(e), north=float(n), crs="EPSG:4326")
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    ppi = con.load_collection("COPERNICUS_PLANT_PHENOLOGY_INDEX", spatial_extent=bbox,
                              temporal_extent=["2022-03-15", "2022-10-31"], bands=["PPI"])
    cty = con.load_collection("CLMS_VLCC_CROP_TYPES_EUROPE_10M_YEARLY_V1", spatial_extent=bbox,
                              temporal_extent=["2022-01-01", "2022-12-31"], bands=["cty"]).reduce_dimension(dimension="t", reducer="max")
    cty = cty.resample_cube_spatial(ppi, method="near")
    cube = None
    for c in CODES:
        sel = (cty.band("cty") == c)
        v = ppi.mask(~sel).rename_labels("bands", [f"ppi_c{c}"])
        cnt = (ppi.band("PPI") * 0 + 1).mask(~sel).add_dimension("bands", f"n_c{c}", type="bands")
        part = v.merge_cubes(cnt)
        cube = part if cube is None else cube.merge_cubes(part)
    agg = cube.aggregate_spatial(geometries=geoms, reducer="mean")
    job = agg.create_job(title="vista s2 test thuringia 2022", out_format="JSON")
    job.start_and_wait()
    res = job.get_results()
    res.download_files(OUT)
    meta = job.describe()
    print("job status", meta.get("status"), "costs", meta.get("costs"), "usage", json.dumps(meta.get("usage", {}))[:400])
    json.dump({"districts": krs[["AGS", "GEN"]].values.tolist(), "job": meta}, open(os.path.join(OUT, "test_meta.json"), "w"), default=str)


if __name__ == "__main__":
    main()
