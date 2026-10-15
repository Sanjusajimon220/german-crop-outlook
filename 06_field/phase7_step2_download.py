"""Field v2 step 2 download (pre-registered 2026-10-14): Sentinel-2 L2A (10 bands + SCL + angles) and Sentinel-1
(VV/VH sigma0, ascending) for pilot cell 38_706, season 2024, 10 m grid EPSG:32632, one GeoTIFF per date."""
import os
import sys

import openeo

OUT = os.path.join("data", "raw", "field_v2", "pilot_2024")
BB = dict(west=304000, south=5648000, east=312000, north=5656000, crs="EPSG:32632")
T = ["2023-10-01", "2024-08-31"]
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL",
         "sunZenithAngles", "viewZenithMean", "sunAzimuthAngles", "viewAzimuthMean"]


def s2(con):
    c = con.load_collection("SENTINEL2_L2A", spatial_extent=BB, temporal_extent=T, bands=BANDS, max_cloud_cover=95)
    c = c.resample_spatial(resolution=10, projection=32632)
    return c.create_job(title="field v2 s2 pilot 2024", out_format="GTiff")


def s1(con, orbit="ascending"):
    c = con.load_collection("SENTINEL1_GRD", spatial_extent=BB, temporal_extent=T, bands=["VV", "VH"],
                            properties={"sat:orbit_state": lambda x: x == orbit}).sar_backscatter(coefficient="sigma0-ellipsoid")
    c = c.resample_spatial(resolution=10, projection=32632)
    title = "field v2 s1 pilot 2024" if orbit == "ascending" else f"field v2 s1 {orbit} pilot 2024"
    return c.create_job(title=title, out_format="GTiff")


def retry(fn, tries=8):
    import time
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if "429" not in str(e) or k == tries - 1:
                raise
            time.sleep(20 * (k + 1))


def job_for(con, title, make):
    """Reuse an existing job with this title (resumable); otherwise create it."""
    for j in con.list_jobs():
        if j.get("title") == title and j.get("status") in ("created", "queued", "running", "finished"):
            return con.job(j["id"])
    return retry(lambda: make(con))


if __name__ == "__main__":
    import time
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    if "descending" in sys.argv:
        jobs = {"s1_desc": job_for(con, "field v2 s1 descending pilot 2024", lambda c: s1(c, "descending"))}
    else:
        jobs = {"s2": job_for(con, "field v2 s2 pilot 2024", s2), "s1": job_for(con, "field v2 s1 pilot 2024", s1)}
    for k, j in jobs.items():
        if retry(j.status) == "created":
            retry(j.start)
        print(k, "job", j.job_id, retry(j.status), flush=True)
    pending = dict(jobs)
    while pending:
        time.sleep(60)
        for k in list(pending):
            st = retry(pending[k].status)
            if st in ("finished", "error", "canceled"):
                n = 0
                if st == "finished":
                    retry(lambda: pending[k].get_results().download_files(os.path.join(OUT, k)))
                    n = len([x for x in os.listdir(os.path.join(OUT, k)) if x.endswith(".tif")])
                print(k, st, "files", n, "credits", retry(pending[k].describe).get("costs"), flush=True)
                del pending[k]
