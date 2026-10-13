"""Field edges without inward shrinking (owner request 2026-10-13): pixel-level Sentinel-2 for a sample of fields.

50 winter wheat fields of 2025 (Rur, >= 2 ha). One openEO job: Sentinel-2 L2A NDVI (SCL 4/5), monthly median for
April, May, June and July 2025, clipped to the fields' bounding box (+30 m), 10 m grid (EPSG:32632) -> GeoTIFF.
Locally, for every 10 m pixel touching a field:
  weight   = share of the pixel area inside the field outline (1 = fully inside; 0 < w < 1 = edge pixel)
  zone     = 'headland' if the pixel centre lies within 20 m of the field border (inside), else 'core'
  field value = weighted mean over ALL pixels (no pixel discarded); core and headland reported separately
Compared with the old method (inward 10 m shrink, plain mean): field means, headland - core difference by month.
Outputs data/processed/pixels/edges_<field>.csv, edges_summary.csv and the GeoTIFF.
"""
import json
import os
import sys
import time

import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

OUT = os.path.join("data", "processed", "pixels")
SUB = 10          # sub-pixel grid per side to compute the area share inside the field


def sample(n=50):
    f = gpd.read_file(os.path.join("data", "processed", "fields_rur_allcrops.gpkg"))
    f = f[(f["VALIDFROM"] == 2025) & (f["CODE"] == 115) & (f["AREA_HA"] >= 2)].sample(n, random_state=3)
    return f[["ID", "AREA_HA", "geometry"]].to_crs(32632).reset_index(drop=True)


def download(f):
    tif = os.path.join(OUT, "ndvi_monthly_2025.tif")
    if os.path.exists(tif):
        return tif
    w, s, e, n = f.total_bounds
    bb = dict(west=float(w - 30), south=float(s - 30), east=float(e + 30), north=float(n + 30), crs="EPSG:32632")
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    s2 = con.load_collection("SENTINEL2_L2A", spatial_extent=bb, temporal_extent=["2025-04-01", "2025-07-31"],
                             bands=["B04", "B08", "SCL"], max_cloud_cover=80)
    scl = s2.band("SCL")
    ndvi = s2.ndvi(nir="B08", red="B04").mask(~((scl == 4) | (scl == 5)))
    m = ndvi.aggregate_temporal_period("month", reducer="median")
    geoms = json.loads(f.assign(geometry=f.buffer(30)).to_crs(4326)[["ID", "geometry"]].to_json())
    m = m.filter_spatial(geoms).resample_spatial(resolution=10, projection=32632)
    job = m.create_job(title="pixels edges 2025", out_format="GTiff")
    job.start_and_wait()
    folder = os.path.join(OUT, "job")
    job.get_results().download_files(folder)
    got = sorted(x for x in os.listdir(folder) if x.endswith(".tif"))
    print("job files:", got, "credits", job.describe().get("costs"), flush=True)
    if len(got) == 1:
        os.replace(os.path.join(folder, got[0]), tif)
        return tif
    return [os.path.join(folder, g) for g in got]


def analyse(f, tif):
    from rasterio.windows import from_bounds
    src = rasterio.open(tif)
    months = ["Apr", "May", "Jun", "Jul"][: src.count]
    f = f.to_crs(src.crs)
    rows = []
    for _, x in f.iterrows():
        g = x.geometry
        win = from_bounds(*g.buffer(30).bounds, transform=src.transform).round_offsets().round_lengths()
        a = src.read(window=win).astype(float)              # only the field's surroundings
        if src.nodata is not None and not np.isnan(src.nodata):
            a[a == src.nodata] = np.nan
        tr = src.window_transform(win)
        hgt, wid = a.shape[1:]
        fine = rasterize([(g, 1)], out_shape=(hgt * SUB, wid * SUB), transform=tr * rasterio.Affine.scale(1 / SUB),
                         fill=0, dtype="uint8")
        share = fine.reshape(hgt, SUB, wid, SUB).mean(axis=(1, 3))          # area share of each pixel inside
        ras = lambda geom: rasterize([(geom, 1)], out_shape=(hgt, wid), transform=tr, fill=0, dtype="uint8")             if not geom.is_empty else np.zeros((hgt, wid), "uint8")
        inner20, inner10 = ras(g.buffer(-20)), ras(g.buffer(-10))
        touch = share > 0
        rec = {"field_id": int(x["ID"]), "area_ha": x["AREA_HA"], "pixels_all": int(touch.sum()),
               "pixels_edge": int(((share > 0) & (share < 1)).sum()), "pixels_shrunk10": int(inner10.sum())}
        for k, mname in enumerate(months):
            v = a[k]
            ok = touch & np.isfinite(v)
            core, head = ok & (inner20 == 1), ok & (inner20 == 0)
            sh = ok & (inner10 == 1)
            rec[f"{mname}_weighted_all"] = float(np.average(v[ok], weights=share[ok])) if ok.any() else np.nan
            rec[f"{mname}_shrunk10"] = float(v[sh].mean()) if sh.any() else np.nan
            rec[f"{mname}_core"] = float(np.average(v[core], weights=share[core])) if core.any() else np.nan
            rec[f"{mname}_headland"] = float(np.average(v[head], weights=share[head])) if head.any() else np.nan
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "edges_fields.csv"), index=False)
    summ = []
    for mname in months:
        summ.append(dict(month=mname, fields=int(d[f"{mname}_weighted_all"].notna().sum()),
                         all_minus_shrunk=round((d[f"{mname}_weighted_all"] - d[f"{mname}_shrunk10"]).median(), 4),
                         headland_minus_core=round((d[f"{mname}_headland"] - d[f"{mname}_core"]).median(), 4),
                         headland_lower_pct=round(100 * ((d[f"{mname}_headland"] < d[f"{mname}_core"]).mean()), 1)))
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT, "edges_summary.csv"), index=False)
    print(f"fields {len(d)}; pixels per field median {d['pixels_all'].median():.0f} (edge {d['pixels_edge'].median():.0f}, "
          f"left after 10 m shrink {d['pixels_shrunk10'].median():.0f})")
    print(s.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    f = sample()
    tif = download(f)
    if isinstance(tif, list):
        print("several files returned; analysis expects one multi-band GeoTIFF:", tif)
        sys.exit()
    analyse(f, tif)
