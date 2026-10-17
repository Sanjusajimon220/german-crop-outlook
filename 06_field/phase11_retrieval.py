"""Leaf retrieval (Sentinel-2 + Landsat) for given area-seasons with fv2 (project log 2026-10-16).
Run `python phase11_retrieval.py AREA YEAR [AREA YEAR ...]`."""
import sys
import time

import fv2

if __name__ == "__main__":
    a = sys.argv[1:]
    for i in range(0, len(a), 2):
        area, year = a[i], int(a[i + 1])
        px = fv2.pixels(area, year)
        t = time.time()
        n2 = fv2.retrieval_s2(area, year, px)
        nl = fv2.retrieval_landsat(area, year, px)
        print(f"{time.strftime('%H:%M')} {area} {year}: S2 dates {n2}, Landsat dates {nl} ({time.time() - t:.0f} s)", flush=True)
