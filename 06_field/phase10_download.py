"""Multi-year / multi-area download (pre-registered docs/project_log.md 2026-10-16): Sentinel-2 L2A (10 bands + SCL +
angles) and Sentinel-1 GRD sigma0 VV/VH, ascending and descending, per area and season, same jobs as pilot 2024.
Run `python phase10_download.py [--s2-only] AREA YEAR [AREA YEAR ...]`; resumable (jobs found again by title).
--s2-only: Sentinel-2 only (held-out water test 2026-11-06 needs no radar)."""
import os
import sys
import time

import openeo

from areas import bounds, period, raw_dir
from phase7_step2_download import BANDS, retry


FOLDERS = ("s2", "s1", "s1_desc")


def pending(area, year, folders=FOLDERS):
    """Product folders (in the given choice) without a .complete marker."""
    return [f for f in folders if not os.path.exists(os.path.join(raw_dir(area, year), f, ".complete"))]


def jobs_for(con, area, year, folders=FOLDERS):
    x0, y0, x1, y1 = bounds(area)
    BB = dict(west=x0, south=y0, east=x1, north=y1, crs="EPSG:32632")
    T = period(year)

    def s2(c):
        cube = c.load_collection("SENTINEL2_L2A", spatial_extent=BB, temporal_extent=T, bands=BANDS, max_cloud_cover=95)
        return cube.resample_spatial(resolution=10, projection=32632).create_job(title=f"field v2 s2 {area} {year}", out_format="GTiff")

    def s1(orbit):
        def make(c):
            cube = c.load_collection("SENTINEL1_GRD", spatial_extent=BB, temporal_extent=T, bands=["VV", "VH"],
                                     properties={"sat:orbit_state": lambda x: x == orbit}).sar_backscatter(coefficient="sigma0-ellipsoid")
            return cube.resample_spatial(resolution=10, projection=32632).create_job(title=f"field v2 s1 {orbit} {area} {year}", out_format="GTiff")
        return make

    existing = {j.get("title"): j for j in retry(con.list_jobs) if j.get("status") in ("created", "queued", "running", "finished")}
    out = {}
    todo = pending(area, year, folders)                            # fully downloaded folders carry a .complete marker
    for folder, title, make in (("s2", f"field v2 s2 {area} {year}", s2), ("s1", f"field v2 s1 ascending {area} {year}", s1("ascending")),
                                ("s1_desc", f"field v2 s1 descending {area} {year}", s1("descending"))):
        if folder not in todo:
            continue
        out[folder] = con.job(existing[title]["id"]) if title in existing else retry(lambda: make(con))
    return out


def main(pairs, folders=FOLDERS):
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    for area, year in pairs:
        jobs = jobs_for(con, area, year, folders)
        for k, j in jobs.items():
            if retry(j.status) == "created":
                retry(j.start)
            print(f"{time.strftime('%H:%M')} {area} {year} {k} job {j.job_id} {retry(j.status)}", flush=True)
        pending = dict(jobs)
        while pending:
            time.sleep(60)
            for k in list(pending):
                st = retry(pending[k].status)
                if st in ("finished", "error", "canceled"):
                    n = 0
                    if st == "finished":
                        d = os.path.join(raw_dir(area, year), k)
                        retry(lambda: pending[k].get_results().download_files(d))
                        n = len([x for x in os.listdir(d) if x.endswith(".tif")])
                        open(os.path.join(d, ".complete"), "w").write(f"{n} files" + chr(10))
                    print(f"{time.strftime('%H:%M')} {area} {year} {k} {st} files {n} credits "
                          f"{retry(pending[k].describe).get('costs')}", flush=True)
                    del pending[k]


if __name__ == "__main__":
    a = sys.argv[1:]
    f = ("s2",) if "--s2-only" in a else FOLDERS
    a = [x for x in a if x != "--s2-only"]
    main([(a[i], int(a[i + 1])) for i in range(0, len(a), 2)], f)
