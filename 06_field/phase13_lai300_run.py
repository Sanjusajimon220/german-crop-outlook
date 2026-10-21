"""LAI 300 m v2 download via the CLMS API (owner OK 2026-10-20): one request per calendar month 2020-10..2025-08, box BOX.

Resumable: state in data/raw/hrvpp_portal/lai300/tasks.json (month -> task id, status, files). Finished tasks are downloaded
(zip -> unzipped into <month>/), checked (opens as GeoTIFF), marked '.complete'. Run detached from D:\vista; log lai300_log.txt.
"""
import calendar
import json
import os
import sys
import time
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase13_clms_api as C

OUT = r"D:\vista\data\raw\hrvpp_portal\lai300"
STATE = os.path.join(OUT, "tasks.json")
LOG = os.path.join(OUT, "lai300_log.txt")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M ") + msg + "\n")


def months(a="2020-10", b="2025-08"):
    y, m = map(int, a.split("-")); out = []
    while f"{y:04d}-{m:02d}" <= b:
        out.append((f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def urls(task):
    found = []
    def walk(v):
        if isinstance(v, dict): [walk(x) for x in v.values()]
        elif isinstance(v, list): [walk(x) for x in v]
        elif isinstance(v, str) and v.startswith("http") and "sdi.eea.europa.eu" not in v: found.append(v)
    walk({k: v for k, v in task.items() if k != "Datasets"})
    return found


def fetch(url, month):
    d = os.path.join(OUT, month); os.makedirs(d, exist_ok=True)
    part = os.path.join(d, "download.part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20): f.write(chunk)
    if zipfile.is_zipfile(part):
        with zipfile.ZipFile(part) as z:
            for n in z.namelist():                                       # flat, no path tricks
                if n.endswith("/"): continue
                with z.open(n) as src, open(os.path.join(d, os.path.basename(n)), "wb") as dst: dst.write(src.read())
        os.remove(part)                                                  # our own temporary zip, contents kept
    else:
        os.replace(part, os.path.join(d, os.path.basename(url.split("?")[0]) or "file.bin"))
    tifs = [f for f in os.listdir(d) if f.lower().endswith((".tif", ".tiff"))]
    import rasterio
    for t in tifs:
        with rasterio.open(os.path.join(d, t)) as ds: ds.read(1)
    open(os.path.join(d, ".complete"), "w").write(f"{len(tifs)} tif\n")
    return len(tifs), sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))


def main():
    os.makedirs(OUT, exist_ok=True)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"2021-04": {"task": "48214122759", "status": "Queued"}}
    s, t0 = C.session(), time.time()
    for month, a, b in months():
        if month not in st:
            try:
                st[month] = {"task": C.request_lai300(s, a, b)["TaskIds"][0]["TaskID"], "status": "Queued"}
            except Exception as e:
                log(f"request {month} failed: {str(e)[:200]}"); continue
            json.dump(st, open(STATE, "w"), indent=1)
    log(f"{len(st)} monthly requests queued")
    while True:
        if time.time() - t0 > 3000: s, t0 = C.session(), time.time()     # access token lasts 1 h
        try:
            tasks = s.get(f"{C.API}/@datarequest_search", timeout=180).json()
        except Exception as e:
            log(f"status check failed: {type(e).__name__}"); time.sleep(120); continue
        for month, v in sorted(st.items()):
            if os.path.exists(os.path.join(OUT, month, ".complete")): continue
            t = tasks.get(v["task"], {}); new = t.get("Status", v["status"])
            if new != v["status"]: log(f"{month}: {new}"); v["status"] = new
            if new == "Finished_ok":
                try:
                    u = urls(t); n, size = fetch(u[0], month); log(f"{month}: {n} GeoTIFFs, {size / 1e6:.1f} MB")
                except Exception as e:
                    log(f"{month}: download/check failed: {type(e).__name__}: {str(e)[:200]}")
                    json.dump(t, open(os.path.join(OUT, f"task_{month}.json"), "w"), indent=1, default=str)
            elif new in ("Finished_nok", "Rejected", "Cancelled"):
                log(f"{month}: {new}; re-requesting")
                m = [x for x in months() if x[0] == month][0]
                try: v.update(task=C.request_lai300(s, m[1], m[2])["TaskIds"][0]["TaskID"], status="Queued")
                except Exception as e: log(f"re-request {month} failed: {str(e)[:200]}")
        json.dump(st, open(STATE, "w"), indent=1)
        done = sum(os.path.exists(os.path.join(OUT, m, ".complete")) for m in st)
        if done == len(months()):
            log("LAI300 DONE"); return
        time.sleep(120)


if __name__ == "__main__":
    main()
