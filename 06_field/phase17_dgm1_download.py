"""NRW DGM1 (1 m elevation, Geobasis NRW, Datenlizenz Deutschland Zero 2.0) for Module A terrain checks (owner OK 2026-10-21).

1 km tiles covering each 8 x 8 km study area plus a 1 km ring (for slope / wetness context). `list` prints count and size;
`get` downloads into data/raw/dgm1_nrw/ (resumable, size-checked).
"""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from areas import AREAS, SIZE

BASE = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_tiff/dgm1_tiff/"
OUT = r"D:\vista\data\raw\dgm1_nrw"
RING = 1


def wanted():
    keys = set()
    for x0, y0 in AREAS.values():
        for e in range(x0 // 1000 - RING, (x0 + SIZE) // 1000 + RING):
            for n in range(y0 // 1000 - RING, (y0 + SIZE) // 1000 + RING):
                keys.add((e, n))
    return keys


def tiles():
    idx = requests.get(BASE + "index.json", timeout=120).json()
    files = idx["datasets"][0]["files"] if "datasets" in idx else idx["files"]
    keys, out = wanted(), []
    for f in files:
        p = f["name"].split("_")
        if len(p) > 4 and p[0] == "dgm1" and (int(p[2]), int(p[3])) in keys:
            out.append((f["name"], int(f.get("size", 0))))
    return out, keys


if __name__ == "__main__":
    t, keys = tiles()
    print(f"{len(t)} files for {len(keys)} wanted tiles, {sum(s for _, s in t) / 1e6:.0f} MB")
    if sys.argv[1:2] == ["get"]:
        os.makedirs(OUT, exist_ok=True)
        for i, (name, size) in enumerate(t):
            dst = os.path.join(OUT, name)
            if os.path.exists(dst) and (not size or os.path.getsize(dst) == size):
                continue
            with requests.get(BASE + name, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dst + ".part", "wb") as f:
                    for c in r.iter_content(1 << 20): f.write(c)
            os.replace(dst + ".part", dst)
            if i % 50 == 0: print(i, name, flush=True)
        print("DGM1 DONE", len(os.listdir(OUT)), "files")
