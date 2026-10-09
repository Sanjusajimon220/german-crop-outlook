"""Phase 4, step 1: Sentinel-2 field time series from the Copernicus Data Space (openEO).

For a set of NRW fields (subsidy records, data/processed/fields_rur.gpkg) the Copernicus servers
compute, for every Sentinel-2 L2A acquisition, the mean surface reflectance of each field:
  bands   B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 (10/20 m; 20 m bands resampled to 10 m)
  angles  sunZenithAngles, viewZenithMean, sunAzimuthAngles, viewAzimuthMean (needed by PROSAIL)
  mask    pixels kept only if the scene classification (SCL) is vegetation (4) or bare soil (5):
          clouds, cloud shadows, cirrus, snow and water are removed before averaging
Field outlines are shrunk 10 m inwards so mixed edge pixels are left out. Only small tables
travel: one row per field and date.

Login: openEO browser login (device code) once; the token is kept in the user's profile.

Run `python phase4_s2_fields.py test` (20 wheat fields, season 2022/23) or
`python phase4_s2_fields.py <crop code> <harvest year> [max fields]` or
`python phase4_s2_fields.py <crop code> all` (harvest years 2021-2026, finished years skipped).
Writes data/processed/s2/s2_<code>_<year>.csv (+ the field GeoJSON used).
"""
import json
import os
import subprocess
import sys

import openeo
import pandas as pd

P = os.path.join("data", "processed")
OUT = os.path.join(P, "s2")
OGR = os.path.join("C:/", "Program Files", "QGIS 3.36.2", "bin", "ogr2ogr.exe")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ANGLES = ["sunZenithAngles", "viewZenithMean", "sunAzimuthAngles", "viewAzimuthMean"]


def fields_geojson(code, year, limit, min_ha=0.3):
    """Fields of one crop and year (>= 0.3 ha, small farms included) as GeoJSON (WGS84, 6 decimals).
    Each outline is shrunk 10 m inwards to drop mixed edge pixels; where that leaves less than
    0.1 ha (10 pixels of 10 m), only 5 m; fields with less than 0.05 ha (5 pixels) left are too
    narrow for Sentinel-2 and skipped. Outlines are simplified to 2 m. Properties: ID, AREA_HA,
    shrink (m) and inner_ha (area left for averaging) - used later as a quality weight."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"fields_{code}_{year}.geojson")
    b10, b5 = "ST_Buffer(geom, -10)", "ST_Buffer(geom, -5)"
    sql = (f"SELECT ID, AREA_HA, shrink, ST_Area(g) / 10000.0 AS inner_ha, "
           f"ST_SimplifyPreserveTopology(g, 2) AS geom FROM ("
           f"SELECT ID, AREA_HA, CASE WHEN ST_Area({b10}) >= 1000 THEN 10 ELSE 5 END AS shrink, "
           f"CASE WHEN ST_Area({b10}) >= 1000 THEN {b10} ELSE {b5} END AS g "
           f"FROM fields WHERE CODE = {code} AND VALIDFROM = {year} AND AREA_HA >= {min_ha}) "
           f"WHERE g IS NOT NULL AND ST_Area(g) >= 500 ORDER BY ID LIMIT {limit}")
    if os.path.exists(path):
        os.remove(path)
    subprocess.run([OGR, "-f", "GeoJSON", "-t_srs", "EPSG:4326", "-lco", "COORDINATE_PRECISION=6",
                    "-dialect", "SQLITE", "-sql", sql, path, os.path.join(P, "fields_rur.gpkg")], check=True)
    gj = json.load(open(path))
    feats = []
    for f in gj["features"]:
        g = f["geometry"]
        if g and g["type"] == "GeometryCollection":     # shrinking can split a field into pieces
            polys = [p["coordinates"] for p in g["geometries"] if p["type"] == "Polygon"]
            polys += [c for p in g["geometries"] if p["type"] == "MultiPolygon" for c in p["coordinates"]]
            g = {"type": "MultiPolygon", "coordinates": polys} if polys else None
        if g and g["type"] in ("Polygon", "MultiPolygon"):
            f["geometry"] = g
            feats.append(f)
    gj["features"] = feats
    return gj, path


def submit_chunk(con, gj, code, year, part):
    """Start the server job for one chunk (or find it if it already exists); returns the job."""
    title = f"field S2 {code} {year} part {part}"
    old = [j for j in con.list_jobs() if j.get("title") == title and j.get("status") in
           ("finished", "running", "queued", "created")]
    if old:
        job = con.job(old[0]["id"])
        if old[0]["status"] == "created":
            job.start()
        return job
    cube = con.load_collection("SENTINEL2_L2A", temporal_extent=[f"{year - 1}-10-01", f"{year}-08-31"],
                               bands=BANDS + ANGLES + ["SCL"], max_cloud_cover=80)
    scl = cube.band("SCL")
    keep = (scl == 4) | (scl == 5)
    cube = cube.filter_bands(BANDS + ANGLES).mask(~keep)
    job = cube.aggregate_spatial(geometries=gj, reducer="mean").create_job(title=title, out_format="CSV")
    job.start()
    return job


def collect_chunk(job, gj, code, year, part):
    out = os.path.join(OUT, f"s2_{code}_{year}_part{part:02d}.csv")
    folder = os.path.join(OUT, f"job_{code}_{year}_part{part:02d}")
    if os.path.exists(folder):
        import shutil
        shutil.rmtree(folder)
    job.get_results().download_files(folder)
    df = pd.concat([pd.read_csv(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".csv")])
    ids = [f["properties"]["ID"] for f in gj["features"]]
    df["field_id"] = df["feature_index"].map(dict(enumerate(ids)))
    props = {f["properties"]["ID"]: f["properties"] for f in gj["features"]}
    for k in ("AREA_HA", "shrink", "inner_ha"):
        df[k.lower()] = df["field_id"].map(lambda i: props[i][k])
    df.to_csv(out, index=False)
    return df


def run(code, year, limit, chunk=1500):
    """All chunks of a season are submitted at once (the server queues them), then collected as
    they finish. Safe to restart: finished chunks on disk are kept, server jobs are reused."""
    import time
    done = os.path.join(OUT, f"s2_{code}_{year}.csv")
    if limit >= 100000 and os.path.exists(done):
        print(f"{done} exists - skipped")
        return None
    gj, path = fields_geojson(code, year, limit)
    feats = gj["features"]
    chunks = {a // chunk: {"type": "FeatureCollection", "features": feats[a:a + chunk]}
              for a in range(0, len(feats), chunk)}
    print(f"{len(feats)} fields (crop code {code}, {year}); {len(chunks)} jobs", flush=True)
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    parts, jobs, tries = {}, {}, {}
    for k, g in chunks.items():
        out = os.path.join(OUT, f"s2_{code}_{year}_part{k:02d}.csv")
        if os.path.exists(out):
            parts[k] = pd.read_csv(out)
        else:
            jobs[k] = submit_chunk(con, g, code, year, k)
    while jobs:
        for k in list(jobs):
            st = jobs[k].status()
            if st == "finished":
                parts[k] = collect_chunk(jobs[k], chunks[k], code, year, k)
                print(f"  part {k}: {len(parts[k])} rows; cpu {jobs[k].describe().get('usage', {}).get('cpu')}",
                      flush=True)
                del jobs[k]
            elif st in ("error", "canceled"):
                tries[k] = tries.get(k, 0) + 1
                if tries[k] > 2:
                    print(f"  part {k}: {st} again - giving up on this chunk for now", flush=True)
                    del jobs[k]
                    continue
                print(f"  part {k}: {st} - resubmitting", flush=True)
                jobs[k] = submit_chunk(con, chunks[k], code, year, k)
        if jobs:
            time.sleep(60)
    if len(parts) < len(chunks):
        print(f"{year}: {len(chunks) - len(parts)} chunk(s) missing - season file not written", flush=True)
        return None
    df = pd.concat([parts[k] for k in sorted(parts)], ignore_index=True)
    df.to_csv(done, index=False)
    v = df.dropna(subset=["B04"])
    print(f"{year}: {len(df)} rows, {df['field_id'].nunique()} fields, {df['date'].nunique()} dates, "
          f"median cloud-free obs per field {v.groupby('field_id').size().median():.0f}", flush=True)
    return df

if __name__ == "__main__":
    if sys.argv[1] == "test":
        run(115, 2023, 20)
    elif sys.argv[2] == "all":
        for yr in range(2021, 2027):
            run(int(sys.argv[1]), yr, 100000)
    else:
        run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 100000)
