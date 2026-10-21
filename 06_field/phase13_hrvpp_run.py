r"""HR-VPP download from WEkEO (owner OK 2026-10-20): tile 31UGS only (holds all 4 study areas), PPI + QFLAG seasonal
trajectories 2020-10..2024 and VPP season-1 SOSD / MAXD / EOSD / QFLAG 2021-2024.

Each file is downloaded to _tmp/, cut to every study area (+200 m, native 31UGS grid, no resampling) into <area>/<id>.tif,
then the temporary full tile is deleted (only 1-2 GB kept; 33 GB free). Resumable: an area cut that exists is skipped.
Token: read from credentials/wekeo.json on every request; on 401 the run waits and re-reads the file (owner pastes a fresh
token). Run detached from D:\vista; log data/raw/hrvpp_portal/hrvpp/hrvpp_log.txt. `test` = first file only.
"""
import os
import sys
import time

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import areas as A
import phase13_wekeo as W

OUT = r"D:\vista\data\raw\hrvpp_portal\hrvpp"
TMP = os.path.join(OUT, "_tmp")
LOG = os.path.join(OUT, "hrvpp_log.txt")
TILE = "T31UGS"
MARGIN = 200
PRODUCTS = [("EO:EEA:DAT:CLMS_HRVPP_ST", {"productType": "PPI"}), ("EO:EEA:DAT:CLMS_HRVPP_ST", {"productType": "QFLAG"})] + \
           [("EO:EEA:DAT:CLMS_HRVPP_VPP", {"productType": p, "productGroupId": "s1"}) for p in ("SOSD", "MAXD", "EOSD", "QFLAG")]


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M ") + msg + "\n")


def items():
    s, out = W.session(), []
    for ds, ex in PRODUCTS:
        for x in W.search(s, W.query(ds, ex)):
            if TILE in x["id"] and not (ds.endswith("VPP") and "_2020_" in x["id"]):
                out.append((ds, x))
    return out


def call(method, url, **kw):
    """One HDA request; on 401 wait for a fresh token in the file (never shows the token)."""
    while True:
        s = W.session()
        try:
            r = s.request(method, url, timeout=600, **kw)
        except Exception as e:
            log(f"request failed ({type(e).__name__}); retry in 60 s"); time.sleep(60); continue
        if r.status_code == 401:
            if W._CACHE:                                             # username / password: log in again at once
                W._CACHE.clear(); log("token renewed (401)"); continue
            log("TOKEN EXPIRED - waiting for a fresh token in credentials/wekeo.json"); time.sleep(120); continue
        if r.status_code in (429, 500, 502, 503, 504):
            log(f"server {r.status_code}; retry in 120 s"); time.sleep(120); continue
        return r


def download(ds, x):
    r = call("POST", f"{W.HDA}/download", json={"dataset_id": ds, "product_id": x["id"], "location": x["properties"]["location"]})
    r.raise_for_status()
    did = r.json()["download_id"]
    path = os.path.join(TMP, x["id"] + ".tif")
    for _ in range(360):
        r = call("GET", f"{W.HDA}/download/{did}", stream=True)
        if r.status_code == 202:
            time.sleep(10); continue
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 22): f.write(chunk)
        os.replace(path + ".part", path)
        return path
    raise TimeoutError("download not ready after 1 h")


def cut(path, pid):
    with rasterio.open(path) as src:
        t = Transformer.from_crs(25832, src.crs, always_xy=True)
        for a in A.AREAS:
            x0, y0, x1, y1 = A.bounds(a)
            xs, ys = t.transform([x0 - MARGIN, x1 + MARGIN, x0 - MARGIN, x1 + MARGIN], [y0 - MARGIN, y0 - MARGIN, y1 + MARGIN, y1 + MARGIN])
            b = src.bounds
            if not (b.left <= min(xs) and max(xs) <= b.right and b.bottom <= min(ys) and max(ys) <= b.top):
                raise ValueError(f"{a} not fully inside {pid} (tile {tuple(round(v) for v in b)})")
            win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
            data = src.read(window=win)
            prof = src.profile | {"height": data.shape[1], "width": data.shape[2], "transform": src.window_transform(win),
                                  "compress": "deflate", "tiled": False, "driver": "GTiff"}
            prof.pop("blockxsize", None); prof.pop("blockysize", None)
            os.makedirs(os.path.join(OUT, a), exist_ok=True)
            with rasterio.open(os.path.join(OUT, a, pid + ".tif.part"), "w", **prof) as dst:
                dst.write(data)
            os.replace(os.path.join(OUT, a, pid + ".tif.part"), os.path.join(OUT, a, pid + ".tif"))
    return data.shape, src.crs.to_epsg()


def done(pid):
    return all(os.path.exists(os.path.join(OUT, a, pid + ".tif")) for a in A.AREAS)


def main(test=False):
    os.makedirs(TMP, exist_ok=True)
    todo = items()
    log(f"{len(todo)} files for {TILE} ({sum(x['properties'].get('size') or 0 for _, x in todo) / 1e9:.1f} GB to transfer)")
    n = 0
    for ds, x in todo:
        pid = x["id"]
        if done(pid): continue
        try:
            path = download(ds, x)
            shape, epsg = cut(path, pid)
            os.remove(path)                                              # our own temporary full tile; the cuts are kept
            n += 1
            log(f"{pid}: cut {shape} EPSG:{epsg} ({n} new)")
        except Exception as e:
            log(f"{pid}: FAILED {type(e).__name__}: {str(e)[:200]}")
        if test: return
    left = sum(not done(x["id"]) for _, x in todo)
    log(f"HRVPP DONE ({left} missing)" if left == 0 else f"HRVPP pass finished, {left} missing - rerun")


if __name__ == "__main__":
    main(test=sys.argv[1:2] == ["test"])
